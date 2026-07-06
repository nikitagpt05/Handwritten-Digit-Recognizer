import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import numpy as np
import cv2


# --- Preprocessing ---
def preprocess_canvas(surface):
    img3d = pygame.surfarray.array3d(surface)
    img3d = np.transpose(img3d, (1, 0, 2))
    gray = cv2.cvtColor(img3d, cv2.COLOR_RGB2GRAY)

    coords = cv2.findNonZero(gray)
    if coords is None:
        return torch.zeros((1, 1, 28, 28))

    x, y, w, h = cv2.boundingRect(coords)
    scale = 20.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    digit_resized = cv2.resize(gray[y:y + h, x:x + w], (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = digit_resized

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # 6. Convert to PyTorch tensor, scale to [0, 1]
    tensor = torch.from_numpy(canvas).float() / 255.0

    # 7. Normalize to [-1, 1] to perfectly match the training data
    tensor = (tensor - 0.5) / 0.5

    return tensor.unsqueeze(0).unsqueeze(0)


# --- Model Architecture ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Load CNN Model ---
model_cnn = CNN()
model_cnn.load_state_dict(torch.load('mnist_cnn.pth', weights_only=True))
model_cnn.eval()


# --- Pygame Loop ---
# --- Pygame Loop ---
def draw_digit():
    pygame.init()
    window_size = 280
    display_height = window_size + 50
    screen = pygame.display.set_mode((window_size, display_height))
    pygame.display.set_caption("Digit Recognizer (CNN)")
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    drawing = False

    pred_cnn = None
    confidence_cnn = None  # 1. Added variable to hold the percentage

    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

                # Grab the image
                input_tensor = preprocess_canvas(screen)

                # Predict with CNN
                with torch.no_grad():
                    out_cnn = model_cnn(input_tensor)

                    # 2. Convert raw output to probabilities
                    probabilities = F.softmax(out_cnn, dim=1)

                    # 3. Extract the highest probability and its corresponding digit
                    max_prob, predicted_idx = torch.max(probabilities, 1)

                    pred_cnn = predicted_idx.item()
                    confidence_cnn = max_prob.item() * 100  # Convert to percentage

                    print(f"Prediction: {pred_cnn} | Confidence: {confidence_cnn:.2f}%")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0, 0, 0))
                    pred_cnn = None
                    confidence_cnn = None  # 4. Reset on clear

            if event.type == pygame.MOUSEMOTION and drawing:
                if event.pos[1] < window_size:
                    pygame.draw.circle(screen, (255, 255, 255), event.pos, 6)

        # Draw the text area
        pygame.draw.rect(screen, (0, 0, 0), (0, window_size, window_size, 50))

        # Display outcome with accuracy
        if pred_cnn is not None and confidence_cnn is not None:
            # 5. Updated the display text to include the percentage (formatted to 1 decimal place)
            text_cnn = font.render(f"Prediction: {pred_cnn}  ({confidence_cnn:.1f}%)", True, (0, 255, 0))
            screen.blit(text_cnn, (10, window_size + 10))

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    draw_digit()