from fuzzywuzzy import fuzz

def clean_line(line: str) -> str:
    return line.strip().lower()

def compare_lines(ground_truth_file, model_output_file, threshold=70):
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = [clean_line(line) for line in f if line.strip()]
    with open(model_output_file, 'r', encoding='utf-8') as f:
        model_output = [clean_line(line) for line in f if line.strip()]

    print("\n🔍 Line-by-Line Comparison:\n")
    for i, gt in enumerate(ground_truth):
        best_match = ""
        best_score = 0
        for model_line in model_output:
            score = fuzz.ratio(gt, model_line)
            if score > best_score:
                best_score = score
                best_match = model_line

        match_str = f"✅ MATCH {best_score}%" if best_score >= threshold else f"❌ WEAK {best_score}%"
        print(f"{i+1}. GT:   {gt}")
        print(f"   OUT:  {best_match}")
        print(f"   → {match_str}\n")

if __name__ == "__main__":
    compare_lines("sbt_gc.txt", "whisper_output.txt")
