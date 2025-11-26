import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
from src.model import ColorizationCNN
from src.utils import preprocess_image, lab_to_rgb, tensor_to_image

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Recolorisation d'Images",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fonctions de l'Application ---

@st.cache_resource
def load_pytorch_model(model_path='models/colorization_model_final.pth'):
    """Charge le modèle PyTorch pré-entraîné une seule fois."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorizationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def colorize_image(model, device, image_bytes):
    """
    Pré-traite, colorise et post-traite une image.
    """
    # 1. Charger et préparer l'image originale
    original_image = Image.open(image_bytes).convert('RGB')
    original_image = np.array(original_image)
    original_h, original_w, _ = original_image.shape
    
    # 2. Pré-traiter l'image pour le modèle (redimensionner, extraire L)
    # On sauvegarde l'image temporairement pour utiliser la fonction existante
    temp_path = "temp_uploaded_image.png"
    cv2.imwrite(temp_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    L_tensor = preprocess_image(temp_path, target_size=(256, 256)).to(device)

    # 3. Inférence du modèle pour prédire les canaux a et b
    with torch.no_grad():
        ab_channels_pred_tensor = model(L_tensor)

    # 4. Post-traitement (L-Channel Grafting)
    # Convertir l'originale en LAB pour récupérer le canal L haute résolution
    lab_original = cv2.cvtColor(original_image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L_original = lab_original[:, :, 0]

    # Redimensionner les canaux a et b prédits à la taille originale
    ab_pred = tensor_to_image(ab_channels_pred_tensor) # Sortie normalisée [0, 1]
    ab_pred_resized = cv2.resize(ab_pred, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    # Dénormaliser les canaux a et b
    ab_pred_denorm = (ab_pred_resized * 255.0) - 128.0

    # Combiner L original et a,b prédits
    lab_colorized = np.stack([L_original, ab_pred_denorm[:, :, 0], ab_pred_denorm[:, :, 1]], axis=-1)
    
    # Reconvertir en RGB
    colorized_rgb = cv2.cvtColor(lab_colorized.astype(np.float32), cv2.COLOR_LAB2RGB)
    colorized_rgb = (colorized_rgb.clip(0, 1) * 255).astype(np.uint8)

    return original_image, colorized_rgb


# --- Interface Utilisateur (UI) ---

# Barre latérale
with st.sidebar:
    st.image("assets/logo_fstt.png", width=200)
    st.header("Mini-Projet de Vision Artificielle")
    st.markdown("""
    Cette application est une démonstration du projet de recolorisation automatique d'images en noir et blanc en utilisant un réseau de neurones convolutionnel (U-Net).
    """)
    st.markdown("---")
    st.markdown("**Réalisé par :**")
    st.markdown("- AZARKAN Zakariae")
    st.markdown("- BENABDELLAH Badr")
    st.markdown("- BOUBACAR Sangare")
    st.markdown("**Encadré par :** Pr. M'hamed AIT KBIR")

# Contenu principal
st.title("🎨 Recolorisation Automatique d'Images")
st.markdown("Déposez une image en noir et blanc (ou en couleur) pour la voir prendre vie avec des couleurs générées par l'IA.")

uploaded_file = st.file_uploader(
    "Choisissez une image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.info("Image chargée avec succès. Cliquez sur le bouton pour démarrer la colorisation.")
    
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(uploaded_file, caption="Image Originale", use_container_width=True)

    with col2:
        placeholder = st.empty()
        placeholder.image("https://placehold.co/600x400/2A3137/FFFFFF?text=Votre+image+coloris%C3%A9e%0Aappara%C3%AEtra+ici...", caption="Image Colorisée", use_container_width=True)

    if st.button("Coloriser l'Image", type="primary", use_container_width=True):
        model, device = load_pytorch_model()
        
        with st.spinner("La magie opère... Colorisation en cours..."):
            original, colorized = colorize_image(model, device, uploaded_file)
        
        st.success("Colorisation terminée !")
        
        # Mettre à jour la colonne de droite avec le résultat
        placeholder.image(colorized, caption="Image Colorisée", use_container_width=True)
        
        # Bouton de téléchargement
        result_image_bytes = cv2.imencode('.png', cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button(
            label="📥 Télécharger l'image colorisée",
            data=result_image_bytes,
            file_name=f"colorized_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True
        )

else:
    st.info("En attente d'une image à traiter.")
