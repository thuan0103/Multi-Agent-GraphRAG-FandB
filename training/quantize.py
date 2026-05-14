import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def quantize_awq(merged_model_path: str, output_path: str) -> None:
    """
    AWQ quantization — tốt nhất cho vLLM/SGLang serving.
    Accuracy loss thấp hơn GPTQ ở cùng bit width.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info("Starting AWQ quantization (W4A16)...")

    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        merged_model_path,
        trust_remote_code=True,
        safetensors=True,
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    calib_data = [
        "Cho tôi 1 ly cà phê sữa đá",
        "Có gì ngon không?",
        "Wifi mật khẩu gì?",
        "Ừm...",
        "I'd like a black coffee",
        "What do you recommend?",
        "What time do you close?",
        "Hello there",
    ] * 16  

    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output))
    tokenizer.save_pretrained(str(output))
    logger.info(f"AWQ model saved to {output}")


def quantize_gguf(merged_model_path: str, output_path: str, quant_type: str = "Q4_K_M") -> None:
    logger.info(f"Starting GGUF quantization ({quant_type})...")

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    gguf_path = output / f"router-{quant_type}.gguf"

    subprocess.run([
        "python", "-m", "llama_cpp.convert",
        merged_model_path,
        "--outfile", str(gguf_path),
        "--outtype", "f16",
    ], check=True)

    subprocess.run([
        "llama-quantize",
        str(gguf_path),
        str(output / f"router-{quant_type}-quantized.gguf"),
        quant_type,
    ], check=True)

    logger.info(f"GGUF model saved to {output}")


def run_all(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    merged_path = str(Path(cfg["training"]["output_dir"]) / "merged")

    quantize_awq(
        merged_model_path=merged_path,
        output_path="models/router-awq",
    )

    try:
        quantize_gguf(
            merged_model_path=merged_path,
            output_path="models/router-gguf",
            quant_type="Q4_K_M",
        )
    except Exception as e:
        logger.warning(f"GGUF quantization skipped: {e}")