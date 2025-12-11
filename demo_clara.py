#!/usr/bin/env python3
"""
Demo script for CLaRa Stage 3 (End-to-End) model
"""

import torch
from transformers import AutoModel

def main():
    print("=" * 60)
    print("CLaRa Stage 3 (End-to-End) Demo")
    print("=" * 60)

    # Check CUDA availability
    print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model path - use compression-128 variant
    model_path = "./models/clara-e2e/compression-128"
    print(f"\n📦 Loading model from: {model_path}")

    try:
        # Load model
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        print("✓ Model loaded successfully!")

        # Example usage
        print("\n" + "=" * 60)
        print("Example: Question Answering with Document Retrieval")
        print("=" * 60)

        # Example documents
        documents = [[
            "Apple was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
            "The first Apple Computer, known as Apple I, was designed by Wozniak.",
            "Apple's headquarters is located in Cupertino, California.",
            "The company is famous for its iPhone, iPad, and MacBook products.",
            "Tim Cook has been the CEO of Apple since August 24, 2011."
        ]]

        # Example question
        questions = ["Where is Apple headquartered and who founded it?"]

        print(f"\nQuestion: {questions[0]}")
        print(f"Documents: {len(documents[0])} documents")

        # Generate answer with retrieval
        print("\nGenerating answer with document retrieval...")
        output, topk_indices = model.generate_from_questions(
            questions=questions,
            documents=documents,
            max_new_tokens=64
        )

        print(f"\n✓ Answer: {output[0]}")
        print(f"\n✓ Selected document indices: {topk_indices}")
        print(f"  Selected documents:")
        for idx in topk_indices[0]:
            print(f"    - Doc {idx}: {documents[0][idx][:80]}...")

        print("\n" + "=" * 60)
        print("Demo completed successfully! 🎉")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
