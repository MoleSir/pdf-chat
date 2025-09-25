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

    def load_pdfs(self, file_paths: List[str]):
        """加载 PDF 并构建 FAISS 知识库"""
        if not file_paths:
            return "❌ 请先上传 PDF 文件！"

        result = self.kb.process_pdfs(file_paths)

        if result.get("indexed", 0) > 0:
            self.is_kb_ready = True

        summary_lines = []
        for item in result["summary"]:
            if item["status"] == "ok":
                summary_lines.append(f"✅ {item['file']} 处理成功，共 {item['chunks']} 个文本块")
            else:
                summary_lines.append(f"❌ {item['file']} 处理失败: {item['error']}")
        summary_lines.append(
            f"\n📊 本次新增 {result['indexed']} 个文本块，当前总索引 {result['total_indexed']} 个"
        )

        return "\n".join(summary_lines)

    def chat(self, query: str, history: list, top_k: int = 3):
        """RAG 聊天：检索 + 记忆对话"""
        if not self.is_kb_ready or self.kb.faiss_index is None:
            return history + [["用户", query], ["系统", "⚠️ 知识库未就绪，请先上传 PDF"]]

        # 1. 向量检索
        query_emb = self.kb.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.kb.faiss_index.search(query_emb, top_k)

        retrieved_chunks = []
        for idx in indices[0]:
            if idx == -1:
                continue
            chunk_id = self.kb.faiss_id_order_for_index[idx]
            retrieved_chunks.append(self.kb.faiss_contents_map[chunk_id])

        if not retrieved_chunks:
            return history + [[query, "⚠️ 未检索到相关内容"]]

        # 2. 构造上下文
        context = "\n\n".join(retrieved_chunks)
        prompt = f"基于以下内容回答用户问题：\n\n{context}\n\n用户问题：{query}"

        # 3. 调用带 memory 的 LLM
        answer = self.llm.chat(prompt)

        # 4. 把这轮对话加入 history（Gradio Chatbot 用 [user, bot]）
        history.append([query, answer])
        return history

    def reset_chat(self):
        """清空历史 + LLM memory"""
        self.llm.reset()
        return []


def create_interface(app: RAGChatApp):
    with gr.Blocks() as demo:
        gr.Markdown("## 📄 PDF RAG 聊天助手")

        with gr.Tabs():
            # ---------------- Tab 1: PDF 上传 ----------------
            with gr.TabItem("📄 上传 PDF"):
                with gr.Row():
                    file_input = gr.File(
                        label="上传 PDF",
                        type="filepath",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    load_btn = gr.Button("📥 导入并构建知识库")

                load_output = gr.Textbox(label="加载结果", lines=6)

                # 绑定事件
                load_btn.click(fn=app.load_pdfs, inputs=file_input, outputs=load_output)

            # ---------------- Tab 2: 聊天 ----------------
            with gr.TabItem("💬 聊天"):
                chatbot = gr.Chatbot(label="对话窗口", height=500)
                query_input = gr.Textbox(label="请输入问题", placeholder="在这里提问...")
                with gr.Row():
                    send_btn = gr.Button("🚀 发送")
                    reset_btn = gr.Button("🗑️ 清空对话")

                # 绑定事件
                send_btn.click(fn=app.chat, inputs=[query_input, chatbot], outputs=chatbot)
                reset_btn.click(fn=app.reset_chat, outputs=chatbot)

    return demo


if __name__ == "__main__":
    app = RAGChatApp()
    demo = create_interface(app)
    webbrowser.open("http://127.0.0.1:7860")
    demo.launch(server_port=7860, server_name="0.0.0.0", show_error=True)
