# 🧠 Clasificación de Imágenes con Vision Transformer (ViT)

Este proyecto es una aplicación web desarrollada con **Flask** que permite **subir una imagen y clasificarla automáticamente** usando un modelo de **Vision Transformer (ViT)** preentrenado.

El modelo utilizado es **google/vit-base-patch16-224** de Hugging Face, basado en el mecanismo de **self-attention** para analizar imágenes como secuencias de parches.

---

## 🚀 ¿Qué hace la aplicación?

- Permite subir una imagen desde el navegador
- Procesa la imagen con un modelo **Vision Transformer**
- Predice la clase más probable de la imagen
- Muestra:
  - La imagen cargada
  - La clase predicha
  - El nivel de confianza del modelo

---

## 🧠 Modelo utilizado

- **Vision Transformer (ViT)**
- `google/vit-base-patch16-224`
- Preentrenado con **ImageNet**
- Entrada: imágenes de **224×224 px**
- Salida: clasificación en **1000 clases**

---

## 🛠 Tecnologías usadas

- Python
- Flask
- Hugging Face Transformers
- PyTorch
- PIL (Pillow)

---

## ▶️ Cómo ejecutar el proyecto

1. Instalar dependencias:
```bash
pip install flask transformers torch pillow
