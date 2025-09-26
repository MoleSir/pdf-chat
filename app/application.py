import gradio as gr
from typing import List
import webbrowser
from app.knowledgebase import KnowledgeBase
from app.chatllm import ChatMemoryLLM


class RAGChatApp:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.llm = ChatMemoryLLM()
        self.is_kb_ready = False

        # 聊天参数
        self.temperature = 0.7
        self.top_k = 3
        self.search_mode = "hybrid"  # 可选: vector, bm25, hybrid
        self.alpha = 0.7
        self.use_rerank = False
        self.rerank_top_k = 2

    def set_params(self, temperature, top_k, search_mode, alpha, use_rerank, rerank_top_k):
        self.temperature = temperature
        self.top_k = top_k
        self.search_mode = search_mode
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.rerank_top_k = min(rerank_top_k, top_k)  # rerank_top_k 不允许超过 top_k

        if use_rerank:
            self.kb.enable_cross_encoder()
        else:
            self.kb.disable_cross_encoder()

        return (f"✅ 参数已更新:\n"
                f"- temperature={temperature}\n"
                f"- top_k={top_k}\n"
                f"- search_mode={search_mode}\n"
                f"- alpha={alpha}\n"
                f"- rerank={'开启' if use_rerank else '关闭'}, rerank_top_k={self.rerank_top_k}")
    
    def load_pdfs(self, file_paths: List[str], progress=gr.Progress()):
        """加载 PDF 并构建知识库"""
        if not file_paths:
            return "❌ 请先上传 PDF 文件！"

        progress(0, desc="正在读取 PDF 文件...")
        result = self.kb.process_pdfs(file_paths)

        progress(0.8, desc="正在生成索引...")
        if result.get("indexed", 0) > 0:
            self.is_kb_ready = True

        summary_lines = []
        for item in result["summary"]:
            if item["status"] == "ok":
                summary_lines.append(f"✅ {item['file']} 处理成功，共 {item['chunks']} 个文本块")
            else:
                summary_lines.append(f"❌ {item['file']} 处理失败: {item['error']}")
        summary_lines.append(f"\n📊 本次新增 {result['indexed']} 个文本块")

        progress(1.0, desc="完成！")
        return "\n".join(summary_lines)

    def clear_pdfs(self):
        """删除知识库中的 PDF 数据"""
        self.kb.clear()  # 你需要在 kb 里实现 clear() 方法，重置 FAISS 索引
        self.is_kb_ready = False
        return "🗑️ 已清空知识库 PDF 数据"

    def chat(self, query: str, history: list):
        """RAG 聊天：检索 + 记忆对话"""
        if self.is_kb_ready:
            # 如果有 pdf，增加检索
            if self.search_mode == "vector":
                results = self.kb.search_vector(query, self.top_k)
            elif self.search_mode == "bm25":
                results = self.kb.search_bm2(query, self.top_k)
            else:
                results = self.kb.search(query, self.top_k, self.alpha)

            if self.use_rerank and results:
                results = self.kb.rerank(query, results, top_k=self.rerank_top_k)

            retrieved_chunks = []
            for result in results:
                retrieved_chunks.append(result.text)

            context = "\n\n".join(retrieved_chunks)
            prompt = f"基于以下内容回答用户问题：\n\n{context}\n\n用户问题：{query}"
        else:
            prompt = query

        # 调用带 memory 的 LLM
        answer = self.llm.chat(prompt, self.temperature)

        # 把这轮对话加入 history（Gradio Chatbot 用 [user, bot]）
        history.append([query, answer])
        return history, ""

    def reset_chat(self):
        self.llm.reset()
        return []

    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("## 📄 PDF RAG 聊天助手")

            with gr.Tabs():
                with gr.TabItem("💬 聊天"):
                    chatbot = gr.Chatbot(label="对话窗口", height=500)
                    query_input = gr.Textbox(label="请输入问题", placeholder="在这里提问...")
                    with gr.Row():
                        send_btn = gr.Button("发送")
                        reset_btn = gr.Button("清空对话")

                    # 绑定事件
                    send_btn.click(fn=self.chat, inputs=[query_input, chatbot], outputs=[chatbot, query_input])
                    reset_btn.click(fn=self.reset_chat, outputs=chatbot)

                with gr.TabItem("📄 上传 PDF"):
                    gr.Markdown("### 📂 上传并导入 PDF 知识库")

                    file_input = gr.File(
                        label="选择 PDF 文件",
                        type="filepath",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )

                    with gr.Row():
                        load_btn = gr.Button("📥 导入构建知识库", variant="primary")
                        clear_btn = gr.Button("🗑️ 删除知识库", variant="stop")

                    load_output = gr.Markdown()
                    load_btn.click(fn=self.load_pdfs, inputs=file_input, outputs=load_output)
                    clear_btn.click(fn=self.clear_pdfs, inputs=None, outputs=load_output)

                with gr.TabItem("⚙️ 设置"):
                    gr.Markdown("### LLM 与检索参数设置")
                    temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                    top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K")
                    search_mode = gr.Radio(["vector", "bm25", "hybrid"], value="hybrid", label="检索模式")
                    alpha = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Hybrid α")

                    # rerank 设置
                    use_rerank = gr.Checkbox(label="是否启用 Cross-Encoder Rerank", value=False)
                    rerank_top_k = gr.Slider(1, 10, value=2, step=1, label="Rerank Top-K")

                    apply_btn = gr.Button("应用参数", variant="primary")
                    output_box = gr.Textbox(label="参数更新状态", interactive=False)

                    apply_btn.click(
                        fn=self.set_params,
                        inputs=[temperature, top_k, search_mode, alpha, use_rerank, rerank_top_k],
                        outputs=output_box
                    )

        return demo
