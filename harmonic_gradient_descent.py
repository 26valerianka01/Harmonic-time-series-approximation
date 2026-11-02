"""
Нелинейная аппроксимация временного ряда суммой гармоник
методом градиентного спуска (два варианта целевой функции).
Данные: котировки акций TATN (Татнефть).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath="TATN.xlsx"):
    df = pd.read_excel(filepath, engine="openpyxl")
    kw = ["close", "закрыт", "цена", "last", "price"]
    col = next(
        (c for c in df.columns if any(k in str(c).lower() for k in kw)),
        df.select_dtypes(include=[np.number]).columns[0]
    )
    prices = df[col].dropna().values.astype(float)
    t = np.arange(len(prices), dtype=float)
    print(f"Точек: {len(prices)}, мин={prices.min():.2f}, макс={prices.max():.2f}")
    return t, prices

# ══════════════════════════════════════════════════════════════════════════════
#  ИНИЦИАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def init_params(prices, seed=42, randomize=True):
    p_fft = [
        386.6587,
        240.1536, 0.028918, 1.7977,
        132.9282, 0.087290, 1.2207,
         79.9044, 0.116686, 2.1949,
         54.6916, 0.175344, 2.2665,
         37.4374, 0.058389, 0.1515,
    ]

    #p_fft = [
    #    386,
    #    240, 0.03, 1.8,
    #    133, 0.09, 1.2,
    #    80, 0.1, 2.2,
    #    55, 0.18, 2.3,
    #    37.5, 0.06, 0.15,
    #]

    if not randomize:
        return np.array(p_fft, dtype=float)

    # Генерация случайных стартовых точек
    np.random.seed(seed)
    std_prices = prices.std()
    b = np.random.uniform(prices.mean() * 0.9, prices.mean() * 1.1)
    p = [b]
    for _ in range(5):
        A = np.random.uniform(0.5 * std_prices, 1.5 * std_prices)
        omega = np.random.uniform(0.05, 0.3)
        delta = np.random.uniform(0, 2 * np.pi)
        p.extend([A, omega, delta])

    return np.array(p, dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

def model(t, p):
    y = np.full(len(t), p[0], dtype=float)
    for k in range(5):
        y += p[1 + 3*k] * np.cos(p[2 + 3*k] * t + p[3 + 3*k])
    return y

# ══════════════════════════════════════════════════════════════════════════════
#  ВАРИАНТ 1: минимизация E
# ══════════════════════════════════════════════════════════════════════════════

def loss_E(p, t, y):
    return float(np.sum((model(t, p) - y) ** 2))

def grad_E(p, t, y):
    r = model(t, p) - y
    g = np.zeros(16)
    g[0] = 2.0 * r.sum()
    for k in range(5):
        A, w, d = p[1 + 3*k], p[2 + 3*k], p[3 + 3*k]
        phi = w * t + d
        g[1 + 3*k] = 2.0 * (r * np.cos(phi)).sum()
        g[2 + 3*k] = 2.0 * (r * (-A * t * np.sin(phi))).sum()
        g[3 + 3*k] = 2.0 * (r * (-A * np.sin(phi))).sum()
    return g

# ══════════════════════════════════════════════════════════════════════════════
#  ВАРИАНТ 2: минимизация ε
# ══════════════════════════════════════════════════════════════════════════════

def loss_eps(p, t, y):
    return float(np.sum(grad_E(p, t, y) ** 2))

def hess_E(p, t, y):
    r = model(t, p) - y
    N = len(t)
    H = np.zeros((16, 16))
    H[0, 0] = 2.0 * N
    for k in range(5):
        Ak, wk, dk = p[1+3*k], p[2+3*k], p[3+3*k]
        phk = wk * t + dk
        ck, sk = np.cos(phk), np.sin(phk)
        H[0, 1+3*k] = H[1+3*k, 0] = 2.0 * ck.sum()
        H[0, 2+3*k] = H[2+3*k, 0] = 2.0 * (-Ak * t * sk).sum()
        H[0, 3+3*k] = H[3+3*k, 0] = 2.0 * (-Ak * sk).sum()
        for j in range(5):
            Aj, wj, dj = p[1+3*j], p[2+3*j], p[3+3*j]
            phj = wj * t + dj
            cj, sj = np.cos(phj), np.sin(phj)
            s = int(k == j)
            H[1+3*k, 1+3*j] = 2.0 * (ck * cj).sum()
            v = (-s * r * t * sk - ck * Aj * t * sj).sum()
            H[1+3*k, 2+3*j] = H[2+3*j, 1+3*k] = 2.0 * v
            v = (-s * r * sk - ck * Aj * sj).sum()
            H[1+3*k, 3+3*j] = H[3+3*j, 1+3*k] = 2.0 * v
            H[2+3*k, 2+3*j] = 2.0 * (Aj*Ak*t**2*sj*sk - s*r*Ak*t**2*ck).sum()
            v = (Aj*Ak*t*sj*sk - s*r*Ak*t*ck).sum()
            H[2+3*k, 3+3*j] = H[3+3*j, 2+3*k] = 2.0 * v
            H[3+3*k, 3+3*j] = 2.0 * (Aj*Ak*sj*sk - s*r*Ak*ck).sum()
    return H

def grad_eps(p, t, y):
    return 2.0 * (hess_E(p, t, y) @ grad_E(p, t, y))

# ══════════════════════════════════════════════════════════════════════════════
#  ПРОЕКЦИЯ ПАРАМЕТРОВ
# ══════════════════════════════════════════════════════════════════════════════

def separate(p, std_prices, d_omega=0.005):
    q = p.copy()
    omega_min = 2 * np.pi / 300.0
    omega_max = 2 * np.pi / 12.0
    A_max     = 2.0 * std_prices

    for k in range(5):
        q[1 + 3*k] = float(np.clip(q[1 + 3*k], 0.0, A_max))

    idx_w  = [2 + 3*k for k in range(5)]
    ws     = np.clip([q[i] for i in idx_w], omega_min, omega_max)
    order  = np.argsort(ws)
    ws_s   = ws[order]
    for k in range(1, 5):
        if ws_s[k] - ws_s[k-1] < d_omega:
            ws_s[k] = ws_s[k-1] + d_omega
    ws_s = np.clip(ws_s, omega_min, omega_max)
    for k in range(5):
        q[idx_w[order[k]]] = ws_s[k]
    return q

# ══════════════════════════════════════════════════════════════════════════════
#  ГРАДИЕНТНЫЙ СПУСК
# ══════════════════════════════════════════════════════════════════════════════

def gradient_descent(p0, t, y, std_prices, loss_fn, grad_fn,
                     lr=1e-6, epochs=1000, decay=0.995, clip=3.0,
                     schedule="decay", norm_epochs=50, lr_small=1e-6,
                     verbose=False, label=""):
    p      = p0.copy()
    hist   = []
    lr_cur = float(lr)

    for ep in range(epochs):
        g  = grad_fn(p, t, y)
        nm = np.linalg.norm(g)
        if not np.isfinite(nm) or nm < 1e-14:
            hist.append(loss_fn(p, t, y))
            continue
        if nm > clip:
            g  = g / nm * clip
            nm = clip

        in_norm = (schedule == "normalize_then_small" and ep < norm_epochs)
        if in_norm:
            step = lr_cur * (g / nm)
        elif schedule == "normalize_then_small":
            step = lr_small * g
        else:
            step = lr_cur * g

        L_old = loss_fn(p, t, y)
        p_cand = p - step
        L_new  = loss_fn(p_cand, t, y)

        if not np.isfinite(L_new) or L_new > L_old * 5:
            lr_cur *= 0.5
            hist.append(L_old)
            continue

        p = separate(p_cand, std_prices)
        hist.append(loss_fn(p, t, y))

        if schedule == "decay":
            lr_cur *= decay

        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"  [{label}] итер.{ep+1}/{epochs}  loss={hist[-1]:.4e}  lr={lr_cur:.2e}")

    return p, hist

# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ  (те же 4, что были в оригинале)
# ══════════════════════════════════════════════════════════════════════════════

def plot_split(t, prices, split, pE, pEps):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Эксперимент 1: Разделение на обучающую и тестовую выборку",
                 fontsize=14, fontweight="bold")
    for ax, p, title in zip(axes, [pE, pEps],
            ["Вариант 1: минимизация E (аналитический градиент)",
             "Вариант 2: минимизация ε (аналитический градиент)"]):
        ax.plot(t[:split], prices[:split], color="steelblue", lw=1.4, label="Обучающая выборка")
        ax.plot(t[split:], prices[split:], color="darkorange", lw=1.4, label="Тестовая выборка")
        ax.plot(t, model(t, p), color="crimson", lw=2.0, ls="--", label="Предсказание модели")
        ax.axvline(t[split], color="gray", ls=":", lw=1.5, label="Граница обучение/тест")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Время (дни)"); ax.set_ylabel("Цена, руб.")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("result_split.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_future(t, prices, pE, pEps):
    horizon = int(0.2 * len(t))
    t_fut   = np.arange(len(t), len(t) + horizon, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Эксперимент 2: Прогноз будущих значений",
                 fontsize=14, fontweight="bold")
    for ax, p, title in zip(axes, [pE, pEps],
            ["Вариант 1: минимизация E", "Вариант 2: минимизация ε"]):
        pf = model(t_fut, p)
        ax.plot(t, prices, color="steelblue", lw=1.4, label="Исторические данные")
        ax.plot(t, model(t, p), color="crimson", lw=2.0, ls="--", label="Аппроксимация")
        ax.plot(t_fut, pf, color="green", lw=2.0, ls="-.", label="Прогноз")
        ax.axvline(t[-1], color="gray", ls=":", lw=1.5, label="Начало прогноза")
        ax.fill_between(t_fut, pf*0.93, pf*1.07, alpha=0.12, color="green", label="±7%")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Время (дни)"); ax.set_ylabel("Цена, руб.")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("result_future.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_harmonics(t, prices, p, title="Гармоники модели"):
    fig, axes = plt.subplots(6, 1, figsize=(14, 16))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    axes[0].plot(t, prices, color="steelblue", lw=1.2, label="Исходные данные")
    axes[0].plot(t, model(t, p), color="crimson", lw=2, ls="--", label="Сумма гармоник")
    axes[0].set_title("Исходные данные + итоговая модель")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.35)
    for k in range(5):
        A, w, d = p[1+3*k], p[2+3*k], p[3+3*k]
        period  = 2 * np.pi / abs(w) if w != 0 else float("inf")
        axes[k+1].plot(t, A * np.cos(w * t + d), color=f"C{k+1}", lw=1.5)
        axes[k+1].set_title(
            f"Гармоника {k+1}: A={A:.2f},  ω={w:.5f} рад/день"
            f"  (период≈{period:.1f} дн.),  δ={d:.3f}")
        axes[k+1].grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("harmonics.png", dpi=120, bbox_inches="tight")
    plt.show()


def plot_found_models_no_predictions(t, y, std_prices, starts, schedules, epochs=800):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Найденные модели (разные старты и λ), без предсказаний",
                 fontsize=12, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Увеличил количество цветов до 5
    sched_ru = {"Const": "Постоянный", "Decay": "Убывание", "NormSmall": "Норма и малый шаг"}

    for var_idx, (loss_fn, grad_fn, _) in enumerate([
        (loss_E, grad_E, "В1"),
        (loss_eps, grad_eps, "В2")
    ]):
        for si, (lrE, lrEps, opts, slabel) in enumerate(schedules):
            ax = axes[var_idx, si]
            lr = lrE if var_idx == 0 else lrEps
            ax.plot(t, y, "k.", markersize=2, alpha=0.7, label="Данные")
            for ci, p0 in enumerate(starts):
                pf, _ = gradient_descent(
                    p0.copy(), t, y, std_prices, loss_fn, grad_fn,
                    lr=lr, epochs=epochs, clip=3.0, **opts
                )
                ax.plot(t, model(t, pf), color=colors[ci], lw=1.2, label=f"Старт {ci+1}")
            ax.set_title(f"λ: {sched_ru.get(slabel, slabel)}")
            ax.set_xlabel("Время (дни)")
            ax.set_ylabel("Цена, руб.")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.35)

    axes[0, 0].set_ylabel("Вариант 1 (E)\nЦена, руб.")
    axes[1, 0].set_ylabel("Вариант 2 (ε)\nЦена, руб.")
    plt.tight_layout()
    plt.savefig("result_found_models_no_predictions.png", dpi=130, bbox_inches="tight")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    t, prices = load_data("TATN.xlsx")
    std_prices = float(prices.std())

    # Фиксированная точка из FFT-анализа
    p0_fft = init_params(prices, randomize=False)

    # Генерация 4 случайных стартовых точек
    starts = [p0_fft]
    for seed in [0, 123, 456, 789]:
        starts.append(init_params(prices, seed=seed, randomize=True))

    split = int(0.8 * len(t))
    t_tr, y_tr = t[:split], prices[:split]
    t_te, y_te = t[split:], prices[split:]

    schedules = [
        (3e-6, 8e-9,  {"schedule": "constant",  "norm_epochs": 50},                             "Const"),
        (8e-6, 2e-8,  {"schedule": "decay",      "decay": 0.997, "norm_epochs": 50},             "Decay"),
        (1e-4, 5e-7,  {"schedule": "normalize_then_small", "norm_epochs": 50, "lr_small": 8e-9}, "NormSmall"),
    ]

    print("\n--- Найденные модели (разные старты и λ), без предсказаний ---")
    plot_found_models_no_predictions(t_tr, y_tr, std_prices, starts, schedules, epochs=800)

    results = []
    for lrE, lrEps, opts, label in schedules:
        pE,  hE  = gradient_descent(p0_fft.copy(), t_tr, y_tr, std_prices, loss_E,   grad_E,
                                    lr=lrE,   epochs=1000, clip=3.0, label=f"E-{label}",  **opts)
        pEp, hEp = gradient_descent(p0_fft.copy(), t_tr, y_tr, std_prices, loss_eps, grad_eps,
                                    lr=lrEps, epochs=1000, clip=3.0, label=f"ε-{label}", **opts)
        rE  = rmse(y_te, model(t_te, pE))
        rEp = rmse(y_te, model(t_te, pEp))
        results.append((label, pE, pEp, hE, hEp, rE, rEp, lrE, lrEps, opts))
        print(f"  {label}: В1 RMSE={rE:.2f} | В2 RMSE={rEp:.2f}")

    print(f"\n{'Сценарий':<12} | {'Вариант 1':>10} | {'Вариант 2':>10}")
    print("-" * 38)
    for r in results:
        print(f"{r[0]:<12} | {r[5]:>10.2f} | {r[6]:>10.2f}")

    best_E   = min(results, key=lambda x: x[5])
    best_eps = min(results, key=lambda x: x[6])
    print(f"\nЛучший сценарий для Варианта 1 (E):   {best_E[0]}")
    print(f"Лучший сценарий для Варианта 2 (ε):   {best_eps[0]}")

    _, pE,  _,   hE,  _,  _, _, lrE_b,  _,       opts_bE   = best_E
    _, _,   pEps, _, hEps, _, _, _,     lrEps_b,  opts_beps = best_eps

    print(f"В1: RMSE={rmse(y_te, model(t_te, pE)):.2f} | MAE={mae(y_te, model(t_te, pE)):.2f}")
    print(f"В2: RMSE={rmse(y_te, model(t_te, pEps)):.2f} | MAE={mae(y_te, model(t_te, pEps)):.2f}")

    plot_split(t, prices, split, pE, pEps)

    pE_f,  _ = gradient_descent(p0_fft.copy(), t, prices, std_prices, loss_E,   grad_E,
                                 lr=lrE_b,  epochs=1000, clip=3.0,
                                 verbose=True, label="E(full)",  **opts_bE)
    pEps_f, _ = gradient_descent(p0_fft.copy(), t, prices, std_prices, loss_eps, grad_eps,
                                  lr=lrEps_b, epochs=1000, clip=3.0,
                                  verbose=True, label="ε(full)", **opts_beps)

    print(f"\nВ1 (полный ряд): RMSE={rmse(prices, model(t, pE_f)):.2f}")
    print(f"В2 (полный ряд): RMSE={rmse(prices, model(t, pEps_f)):.2f}")

    plot_future(t, prices, pE_f, pEps_f)
    plot_harmonics(t, prices, pE_f,   "Гармоники - Вариант 1 (E)")
    plot_harmonics(t, prices, pEps_f, "Гармоники - Вариант 2 (ε)")

    for name, p in [("Вариант 1 (E)", pE_f), ("Вариант 2 (ε)", pEps_f)]:
        print(f"\n{name}: b={p[0]:.4f}")
        for k in range(5):
            A, w, d = p[1+3*k], p[2+3*k], p[3+3*k]
            print(f"  Г{k+1}: A={A:8.4f}  ω={w:.6f}  δ={d:.4f}  (T≈{2*np.pi/abs(w):.1f} дн.)")

if __name__ == "__main__":
    main()