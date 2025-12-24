import gradio as gr
from deep_translator import GoogleTranslator

# --- 1. The Backend Logic (Updated to use Deep-Translator) ---
def translate_text(text, target_lang):
    if not text:
        return ""
    try:
        # deep-translator is much more stable than googletrans
        translator = GoogleTranslator(source='auto', target=target_lang)
        result = translator.translate(text)
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- 2. Custom CSS (Professional Look) ---
custom_css = """
.container {max-width: 900px; margin: auto; padding-top: 20px;}
h1 {text-align: center; color: #2563eb; font-family: 'Helvetica', sans-serif;}
.subtitle {text-align: center; color: #6b7280; margin-bottom: 30px;}
footer {visibility: hidden}
"""

# --- 3. The Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css, title="CodeAlpha Translator") as demo:
    
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 🌐 GlobalConnect AI", elem_id="title")
        gr.Markdown("<p class='subtitle'>Instant, neural-network powered translation for professional communication.</p>")

        with gr.Group():
            with gr.Row():
                # Left Side: Input
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Source Text", 
                        placeholder="Type or paste your text here...", 
                        lines=5,
                        show_copy_button=True
                    )
                    
                    # --- EXPANDED LANGUAGE LIST ---
                    lang_selector = gr.Dropdown(
                        choices=[
                            ("English", "en"),
                            ("Spanish (Español)", "es"), 
                            ("French (Français)", "fr"), 
                            ("German (Deutsch)", "de"), 
                            ("Italian (Italiano)", "it"), 
                            ("Portuguese (Português)", "pt"), 
                            ("Hindi (हिन्दी)", "hi"), 
                            ("Gujarati (ગુજરાતી)", "gu"), 
                            ("Chinese (Simplified)", "zh-CN"),
                            ("Japanese (日本語)", "ja"), 
                            ("Korean (한국어)", "ko"), 
                            ("Russian (Русский)", "ru"), 
                            ("Arabic (العربية)", "ar"),
                            ("Turkish (Türkçe)", "tr"),
                            ("Dutch (Nederlands)", "nl"),
                            ("Polish (Polski)", "pl"),
                            ("Indonesian (Bahasa)", "id"),
                            ("Vietnamese (Tiếng Việt)", "vi"),
                            ("Thai (ไทย)", "th"),
                            ("Bengali (বাংলা)", "bn")
                        ], 
                        value="fr", 
                        label="Target Language", 
                        info="Select the destination language"
                    )
                    
                    with gr.Row():
                        clear_btn = gr.ClearButton([input_text], variant="secondary")
                        translate_btn = gr.Button("✨ Translate Text", variant="primary", scale=2)

                # Right Side: Output
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Translation Result", 
                        lines=5, 
                        interactive=False, 
                        show_copy_button=True, 
                        placeholder="Translation will appear here..."
                    )

        # Examples
        gr.Examples(
            examples=[
                ["Hello, I am excited to join this internship.", "fr"],
                ["Where can I find the nearest metro station?", "es"],
                ["Artificial Intelligence is transforming the world.", "gu"], 
                ["Technology is growing fast.", "hi"]
            ],
            inputs=[input_text, lang_selector],
            label="Try these examples:"
        )

        gr.Markdown("---")
        gr.Markdown("<div style='text-align: center; color: grey;'>Built by Mann Patel | CodeAlpha Internship 2025</div>")

    # Connect the Logic
    translate_btn.click(fn=translate_text, inputs=[input_text, lang_selector], outputs=output_text)

# Launch
if __name__ == "__main__":
    demo.launch()