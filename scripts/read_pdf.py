import fitz
doc = fitz.open("LLM - multiagent_llm_competency_test.pdf")
print("Total pages:", len(doc))
for i in range(len(doc)):
    t = doc[i].get_text()
    low = t.lower()
    if "c1" in t or "c2" in t or "a3" in t or "fine" in low or "quantiz" in low or "sft" in low or "lora" in low:
        print(f"\n=== PAGE {i+1} ===")
        print(t[:3000].encode("ascii", "replace").decode())
