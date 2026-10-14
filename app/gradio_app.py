import html
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import shap


# =====================================================
# 1. CẤU HÌNH
# =====================================================

BASE_DIR = Path(r"D:\checkk")

DATA_PATH = BASE_DIR / "data" / "diabetes_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
THRESHOLD_PATH = BASE_DIR / "results" / "reports" / "selected_threshold.csv"

TABLE_HEADERS = ["Đặc trưng", "Giá trị", "SHAP", "Ảnh hưởng"]

DEFAULTS = (1, 120, 70, 20, 80, 25, 0.5, 30)


# =====================================================
# 2. TẢI MÔ HÌNH VÀ SHAP
# =====================================================

pipeline = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

threshold_df = pd.read_csv(THRESHOLD_PATH)
SELECTED_THRESHOLD = float(threshold_df["Selected_Threshold"].iloc[0])

scaler = pipeline.named_steps["scaler"]
classifier = pipeline.named_steps["model"]

df = pd.read_csv(DATA_PATH)
X = df[feature_names]

explainer = shap.LinearExplainer(classifier, scaler.transform(X))


# =====================================================
# 3. HÀM XỬ LÝ
# =====================================================

def build_input(*values):
    if any(v is None for v in values):
        raise ValueError("Vui lòng nhập đầy đủ tất cả các chỉ số.")
    return pd.DataFrame([list(values)], columns=feature_names)


def explain_prediction(X_input):
    shap_values = explainer.shap_values(scaler.transform(X_input))[0]

    explanation = pd.DataFrame({
        "Đặc trưng": feature_names,
        "Giá trị": X_input.iloc[0].values,
        "SHAP": shap_values,
    })

    explanation["Ảnh hưởng"] = explanation["SHAP"].apply(
        lambda v: "↑ Tăng nguy cơ" if v > 0
        else "↓ Giảm nguy cơ" if v < 0
        else "— Không đáng kể"
    )

    explanation["_abs"] = explanation["SHAP"].abs()

    explanation = (
        explanation.sort_values("_abs", ascending=False)
        .head(5)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )

    explanation["Giá trị"] = explanation["Giá trị"].round(2)
    explanation["SHAP"] = explanation["SHAP"].round(4)
    return explanation


def render_result(probability=None, error=None):
    """Sinh thẻ kết quả HTML (xác suất + thanh tiến trình + phân loại)."""

    if error is not None:
        return f"""
        <div class="result-card state-error">
            <div class="result-badge">Không thể dự đoán</div>
            <p class="result-note">{html.escape(error)}</p>
        </div>
        """

    if probability is None:
        return """
        <div class="result-card state-idle">
            <div class="result-badge">Chưa có kết quả</div>
            <p class="result-note">
                Nhập các chỉ số sức khỏe rồi nhấn <b>Dự đoán nguy cơ</b>.
            </p>
        </div>
        """

    is_high = probability >= SELECTED_THRESHOLD
    state = "state-high" if is_high else "state-low"
    label = "Nguy cơ cao" if is_high else "Nguy cơ thấp"
    icon = "⚠️" if is_high else "✅"

    return f"""
    <div class="result-card {state}">
        <div class="result-top">
            <div>
                <div class="result-label">Xác suất nguy cơ</div>
                <div class="result-prob">{probability:.1%}</div>
            </div>
            <div class="result-badge">{icon} {label}</div>
        </div>

        <div class="bar">
            <div class="bar-fill" style="width:{probability * 100:.1f}%"></div>
            <div class="bar-threshold"
                 style="left:{SELECTED_THRESHOLD * 100:.1f}%"></div>
        </div>
        <div class="bar-legend">
            <span>0%</span>
            <span>Ngưỡng phân loại: {SELECTED_THRESHOLD:.0%}</span>
            <span>100%</span>
        </div>
    </div>
    """


def predict(*values):
    try:
        X_input = build_input(*values)
        probability = float(pipeline.predict_proba(X_input)[0, 1])
        return render_result(probability), explain_prediction(X_input)

    except Exception as error:
        return (
            render_result(error=str(error)),
            pd.DataFrame(columns=TABLE_HEADERS),
        )


def reset_form():
    return (*DEFAULTS, render_result(), pd.DataFrame(columns=TABLE_HEADERS))


# =====================================================
# 4. GIAO DIỆN
# =====================================================

theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1080px !important; margin: auto !important; }

/* ---------- Header ---------- */
.app-header {
    background: linear-gradient(135deg, #0f766e 0%, #0e7490 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: #fff;
    margin-bottom: 8px;
}
.app-header h1 { margin: 0 0 6px; font-size: 1.7rem; font-weight: 700; color: #fff; }
.app-header p  { margin: 0; opacity: .92; font-size: .98rem; line-height: 1.55; color: #fff; }
.app-header .tag {
    display: inline-block; margin-bottom: 10px; padding: 3px 12px;
    border-radius: 999px; background: rgba(255,255,255,.18);
    font-size: .75rem; letter-spacing: .04em; text-transform: uppercase;
}

/* ---------- Disclaimer ---------- */
.disclaimer {
    background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #f59e0b;
    border-radius: 10px; padding: 12px 16px; color: #78350f;
    font-size: .88rem; line-height: 1.55;
}
.disclaimer b { color: #92400e; }

/* ---------- Section title ---------- */
.section-title {
    font-size: 1rem; font-weight: 600; margin: 0 0 4px;
    padding-bottom: 8px; border-bottom: 2px solid #e2e8f0;
}

/* ---------- Result card ---------- */
.result-card {
    border-radius: 14px; padding: 22px 24px; border: 1px solid;
    transition: all .25s ease;
}
.result-top { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
.result-label { font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; opacity: .75; }
.result-prob  { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.result-badge {
    display: inline-block; padding: 8px 16px; border-radius: 999px;
    font-weight: 600; font-size: .95rem;
}
.result-note { margin: 8px 0 0; font-size: .92rem; }

.bar { position: relative; height: 12px; border-radius: 999px; background: rgba(0,0,0,.08); margin-top: 18px; }
.bar-fill { height: 100%; border-radius: 999px; transition: width .5s ease; }
.bar-threshold {
    position: absolute; top: -5px; width: 2px; height: 22px;
    background: #334155; border-radius: 2px;
}
.bar-legend { display: flex; justify-content: space-between; font-size: .75rem; margin-top: 6px; opacity: .7; }

.state-idle  { background: #f8fafc; border-color: #e2e8f0; color: #475569; }
.state-idle .result-badge { background: #e2e8f0; color: #475569; }

.state-low   { background: #f0fdf4; border-color: #86efac; color: #166534; }
.state-low .result-badge { background: #dcfce7; color: #166534; }
.state-low .bar-fill { background: #22c55e; }

.state-high  { background: #fef2f2; border-color: #fca5a5; color: #991b1b; }
.state-high .result-badge { background: #fee2e2; color: #991b1b; }
.state-high .bar-fill { background: #ef4444; }

.state-error { background: #fff7ed; border-color: #fdba74; color: #9a3412; }
.state-error .result-badge { background: #ffedd5; color: #9a3412; }

/* ---------- Footer ---------- */
.app-footer { text-align: center; font-size: .8rem; opacity: .65; padding: 8px 0 4px; }
"""

HEADER_HTML = """
<div class="app-header">
    <span class="tag">Explainable AI</span>
    <h1>🩺 Hệ thống AI đánh giá nguy cơ tiểu đường</h1>
    <p>
        Kết hợp <b>Machine Learning</b> và <b>SHAP</b> để ước lượng nguy cơ
        và giải thích các yếu tố ảnh hưởng đến dự đoán.
    </p>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer">
    <b>⚠️ Cảnh báo quan trọng:</b> Đây là mô hình AI mô phỏng phục vụ học tập,
    nghiên cứu và minh họa công nghệ. Kết quả chỉ mang tính tham khảo, không có
    giá trị chẩn đoán và không thay thế ý kiến bác sĩ. Không dùng kết quả để tự
    chẩn đoán, điều trị hoặc đưa ra quyết định y tế.
</div>
"""

with gr.Blocks(
    title="Hệ thống AI đánh giá nguy cơ tiểu đường",
    theme=theme,
    css=CSS,
) as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(DISCLAIMER_HTML)

    # ---------- Đầu vào ----------
    with gr.Row(equal_height=True):

        with gr.Column(variant="panel"):
            gr.HTML('<div class="section-title">👤 Thông tin cơ bản</div>')

            with gr.Row():
                pregnancies = gr.Number(
                    label="Số lần mang thai", value=DEFAULTS[0], minimum=0, precision=0
                )
                age = gr.Number(
                    label="Tuổi", value=DEFAULTS[7], minimum=1, precision=0
                )

            with gr.Row():
                bmi = gr.Number(label="BMI (kg/m²)", value=DEFAULTS[5], minimum=0)
                diabetes_pedigree = gr.Number(
                    label="Chỉ số di truyền tiểu đường",
                    value=DEFAULTS[6],
                    minimum=0,
                )

        with gr.Column(variant="panel"):
            gr.HTML('<div class="section-title">🧪 Các chỉ số </div>')

            with gr.Row():
                glucose = gr.Number(
                    label="Glucose (mg/dL)", value=DEFAULTS[1], minimum=0
                )
                insulin = gr.Number(
                    label="Insulin (μU/mL)", value=DEFAULTS[4], minimum=0
                )

            with gr.Row():
                blood_pressure = gr.Number(
                    label="Huyết áp (mm Hg)", value=DEFAULTS[2], minimum=0
                )
                skin_thickness = gr.Number(
                    label="Độ dày da (mm)", value=DEFAULTS[3], minimum=0
                )

    with gr.Row():
        predict_button = gr.Button("🔍 Dự đoán nguy cơ", variant="primary", size="lg", scale=3)
        reset_button = gr.Button("🔄 Đặt lại", variant="secondary", size="lg", scale=1)

    # ---------- Kết quả ----------
    gr.HTML('<div class="section-title">📋 Kết quả đánh giá</div>')

    result_html = gr.HTML(render_result())

    explanation = gr.Dataframe(
        headers=TABLE_HEADERS,
        value=pd.DataFrame(columns=TABLE_HEADERS),
        label="🔎 Top 5 yếu tố ảnh hưởng nhiều nhất đến dự đoán",
        interactive=False,
        wrap=True,
    )

    with gr.Accordion("💡 Cách đọc giá trị SHAP", open=False):
        gr.Markdown(
            """
            - **↑ Tăng nguy cơ**: đặc trưng đẩy dự đoán theo hướng nguy cơ cao hơn.
            - **↓ Giảm nguy cơ**: đặc trưng đẩy dự đoán theo hướng nguy cơ thấp hơn.
            - Giá trị SHAP càng lớn về **độ lớn tuyệt đối** thì ảnh hưởng càng mạnh.

            > SHAP giải thích **hành vi của mô hình**, không chứng minh một yếu tố
            > trực tiếp gây ra bệnh.
            """
        )

    gr.HTML(
        '<div class="app-footer">Hệ thống nghiên cứu / giáo dục — '
        "không phải thiết bị hoặc phần mềm chẩn đoán y tế.</div>"
    )

    # ---------- Sự kiện ----------
    inputs = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]

    predict_button.click(fn=predict, inputs=inputs, outputs=[result_html, explanation])

    reset_button.click(
        fn=reset_form,
        inputs=[],
        outputs=[*inputs, result_html, explanation],
    )


# =====================================================
# 5. CHẠY ỨNG DỤNG
# =====================================================

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)