"""Knowledge Distillation — training a small model from a large model's outputs."""

import numpy as np

def softmax(x, T=1.0):
    e = np.exp((x - x.max(axis=-1, keepdims=True)) / T)
    return e / e.sum(axis=-1, keepdims=True)

def distillation_loss(student_logits, teacher_logits, hard_labels, T=3.0, alpha=0.7):
    """Combined soft (teacher) and hard (label) loss."""
    soft_student = softmax(student_logits, T)
    soft_teacher = softmax(teacher_logits, T)
    # KL divergence (soft targets)
    soft_loss = -np.mean(np.sum(soft_teacher * np.log(soft_student + 1e-8), axis=1))
    # cross-entropy (hard targets)
    hard_student = softmax(student_logits, 1.0)
    hard_loss = -np.mean(np.sum(hard_labels * np.log(hard_student + 1e-8), axis=1))
    return alpha * (T**2) * soft_loss + (1 - alpha) * hard_loss

# --- demo ---
np.random.seed(42)
n_classes = 5

# teacher: large model with sharp predictions
W_teacher = np.random.randn(8, n_classes) * 0.5
X = np.random.randn(20, 8)
teacher_logits = X @ W_teacher

# student: small model
W_student = np.random.randn(8, n_classes) * 0.1
y = teacher_logits.argmax(axis=1)
y_onehot = np.eye(n_classes)[y]

print("=== Knowledge Distillation ===\n")
print(f"Temperature effect on teacher softmax (sample 0):")
for T in [1.0, 3.0, 5.0, 10.0]:
    probs = softmax(teacher_logits[0:1], T)[0]
    print(f"  T={T:>4.1f}: {probs.round(3)}")

# train student with distillation
lr = 0.01
for epoch in range(200):
    student_logits = X @ W_student
    loss = distillation_loss(student_logits, teacher_logits, y_onehot, T=3.0)
    grad = (softmax(student_logits) - y_onehot) / len(X)
    W_student -= lr * X.T @ grad

student_preds = (X @ W_student).argmax(axis=1)
teacher_preds = teacher_logits.argmax(axis=1)
agreement = (student_preds == teacher_preds).mean()
print(f"\nStudent-teacher agreement: {agreement:.1%}")
