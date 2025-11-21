#!/usr/bin/env python3
"""
Point d'entrée principal pour le projet de recolorisation d'images.
"""

import argparse
from src.train import train_model
from src.colorize import colorize_images, demo_colorization
from src.utils import evaluate_results
from src.subjective_evaluation import create_comparison_grid


def main():
    parser = argparse.ArgumentParser(description="Recolorisation d'Images Historiques")
    parser.add_argument("--mode", choices=["train", "colorize", "evaluate", "demo", "complete"], required=True,
                       help="Mode d'exécution")
    parser.add_argument("--input", default="data/historical", 
                       help="Dossier d'images d'entrée")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Taille des batches")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Taux d'apprentissage")
    parser.add_argument("--model", default="models/colorization_model_final.pth",
                       help="Chemin vers le modèle")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🚀 Entraînement du modèle...")
        train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
        
    elif args.mode == "colorize":
        print("🎨 Colorisation des images...")
        colorize_images(args.input, args.model)
        
    elif args.mode == "evaluate":
        print("📊 Évaluation des résultats...")
        evaluate_results()
        
    elif args.mode == "demo":
        print("🎭 Démonstration de colorisation...")
        # Prendre la première image du dossier
        from pathlib import Path
        input_path = Path(args.input)
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        if images:
            demo_colorization(str(images[0]), args.model)
        else:
            print("❌ Aucune image trouvée dans le dossier")
            
    elif args.mode == "complete":
        print("🚀 Évaluation complète du projet...")
        # Coloriser et évaluer
        colorize_images('data/test', args.model)
        evaluate_results()
        create_comparison_grid(9)
        colorize_images('data/historical', args.model)
        print("✅ Évaluation complète terminée !")


if __name__ == "__main__":
    main()
