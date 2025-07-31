from PIL import Image, ImageEnhance, ImageOps
import random
import datetime

# ==== Transformações seguras ====
def rotate(img, magnitude):
    angle = random.uniform(-magnitude, magnitude)
    print(f"  - rotate: {angle:.2f} graus")
    return img.rotate(angle)

def brightness(img, magnitude):
    factor = 1 + (magnitude / 30) * random.choice([-1, 1]) * 0.5
    print(f"  - brightness: fator {factor:.2f}")
    return ImageEnhance.Brightness(img).enhance(factor)

def contrast(img, magnitude):
    factor = 1 + (magnitude / 30) * random.choice([-1, 1]) * 0.5
    print(f"  - contrast: fator {factor:.2f}")
    return ImageEnhance.Contrast(img).enhance(factor)

def color(img, magnitude):
    factor = 1 + (magnitude / 30) * random.choice([-1, 1]) * 0.5
    print(f"  - color: fator {factor:.2f}")
    return ImageEnhance.Color(img).enhance(factor)

def sharpness(img, magnitude):
    factor = 1 + (magnitude / 30) * random.choice([-1, 1]) * 1.0
    print(f"  - sharpness: fator {factor:.2f}")
    return ImageEnhance.Sharpness(img).enhance(factor)

def posterize(img, magnitude):
    bits = 8 - int((magnitude / 30) * 4)
    bits = max(1, bits)
    print(f"  - posterize: {bits} bits")
    return ImageOps.posterize(img, bits)

def solarize(img, magnitude):
    threshold = int((magnitude / 30) * 256)
    print(f"  - solarize: threshold {threshold}")
    return ImageOps.solarize(img, threshold)

def invert(img, _):
    print("  - invert")
    return ImageOps.invert(img)

# Lista das operações possíveis
TRANSFORMACOES = [
    rotate,
    brightness,
    contrast,
    color,
    sharpness,
    posterize,
    solarize,
    invert,
]

# RandAugment seguro com log
def rand_augment_seguro(img, N=4, M=25, nome="imagem"):
    print(f"\nAplicando RandAugment em {nome}:")
    ops = random.sample(TRANSFORMACOES, N)
    for op in ops:
        img = op(img, M)
    return img

# ========== Execução ==========
if __name__ == "__main__":
    caminho_imagem = "C:/Users/Casa/Desktop/codigos-pibic2/codigos-pibic/img_test.jpg"
    imagem_original = Image.open(caminho_imagem).convert("RGB")

    # Aplica duas vezes e loga os detalhes
    aug1 = rand_augment_seguro(imagem_original.copy(), N=5, M=15, nome="aug1")
    aug2 = rand_augment_seguro(imagem_original.copy(), N=5, M=15, nome="aug2")

    # Salvar com timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    aug1_path = f"aug1_seguro_{timestamp}.jpg"
    aug2_path = f"aug2_seguro_{timestamp}.jpg"

    aug1.save(aug1_path)
    aug2.save(aug2_path)

    # Exibe as imagens
    aug1.show(title="Augmentada 1")
    aug2.show(title="Augmentada 2")

    print(f"\nImagens salvas:\n  - {aug1_path}\n  - {aug2_path}")
