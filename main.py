#!/usr/bin/env python3
"""
Point d'entrée principal pour le projet de recolorisation d'images.
"""

import argparse
from src.train import train_model
from src.colorize import colorize_images
from src.utils import evaluate_results


def main():
    parser = argparse.ArgumentParser(description="Recolorisation d'Images")
    parser.add_argument("--mode", choices=["train", "colorize", "evaluate"], required=True)
    parser.add_argument("--input", default="data/historical")
    parser.add_argument("--epochs", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🚀 Entraînement du modèle...")
        train_model(epochs=args.epochs)
        
    elif args.mode == "colorize":
        print("🎨 Colorisation des images...")
        colorize_images(args.input)
        
    elif args.mode == "evaluate":
        print("📊 Évaluation des résultats...")
        evaluate_results()


if __name__ == "__main__":
    main()
