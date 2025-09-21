import os
import re
import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional

# LangChain imports
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

# AWS Bedrock imports
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock

# Configuration
VECTORSTORE_DIR = "./vectorstore"
MD_PATHS = ["./data/ベンダー調査.md"]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5

def get_embeddings():
    """Get AWS Bedrock embeddings instance"""
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )

def get_llm():
    """Get AWS Bedrock LLM instance"""
    return ChatBedrock(
        model_id="anthropic.claude-v2",
        region_name="us-east-1",
        model_kwargs={"temperature": 0}
    )

def load_md(md_paths: List[str]) -> List[Document]:
    """Load markdown files and return documents"""
    documents = []
    
    for md_path in md_paths:
        if os.path.exists(md_path):
            try:
                loader = TextLoader(md_path, encoding='utf-8')
                docs = loader.load()
                # Add metadata
                for i, doc in enumerate(docs):
                    doc.metadata = {
                        "source": "vendors_md",
                        "chunk_id": i,
                        "file_path": md_path
                    }
                documents.extend(docs)
                st.success(f"✅ 読み込み完了: {md_path}")
                break
            except Exception as e:
                st.error(f"❌ 読み込みエラー {md_path}: {str(e)}")
        else:
            st.warning(f"⚠️ ファイルが見つかりません: {md_path}")
    
    if not documents:
        st.error("❌ ベンダー調査.mdファイルが見つかりません。以下の場所に配置してください：")
        st.code("\n".join(MD_PATHS))
        return []
    
    return documents

def build_or_load_vectorstore(docs: List[Document], persist_dir: str) -> FAISS:
    """Build or load FAISS vectorstore"""
    persist_path = Path(persist_dir)
    
    # Check if vectorstore exists
    if persist_path.exists() and any(persist_path.iterdir()):
        try:
            st.info("🔄 既存のベクトルストアを読み込み中...")
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(persist_dir, embeddings)
            st.success("✅ ベクトルストアの読み込み完了")
            return vectorstore
        except Exception as e:
            st.warning(f"⚠️ ベクトルストアの読み込みに失敗: {str(e)}")
            st.info("🔄 新規作成します...")
    
    # Build new vectorstore
    if not docs:
        raise ValueError("No documents to build vectorstore")
    
    st.info("🔄 ベクトルストアを新規作成中...")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings and vectorstore
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save vectorstore
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    st.success(f"✅ ベクトルストア作成完了（{len(splits)}チャンク）")
    
    return vectorstore

def make_chain(vectorstore: FAISS, k: int) -> RetrievalQA:
    """Create RAG chain"""
    llm = get_llm()
    
    # Custom prompt template
    prompt_template = """あなたはAIベンダー調査の専門家です。以下の情報を基に、ユーザーの質問に答えてください。

文脈情報:
{context}

質問: {question}

回答は以下の形式でお願いします：
1. 質問に対する要約回答
2. 関連するベンダー情報（ベンダー名 / 強み / URL）を箇条書きで
3. 最後に「参照ソース:」として、根拠となった情報の抜粋を簡潔に記載

回答:"""

    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return chain

def extract_vendor_info(text: str) -> Dict[str, str]:
    """Extract vendor information from text using regex"""
    info = {"name": "", "strength": "", "url": ""}
    
    # Extract vendor name
    name_match = re.search(r'### ベンダー \d+: ([^｜]+)', text)
    if name_match:
        info["name"] = name_match.group(1).strip()
    
    # Extract strength
    strength_match = re.search(r'強み: ([^｜]+)', text)
    if strength_match:
        info["strength"] = strength_match.group(1).strip()
    
    # Extract URL
    url_match = re.search(r'URL: (https?://[^\s｜]+)', text)
    if url_match:
        info["url"] = url_match.group(1).strip()
    
    return info

def display_vendor_cards(source_docs: List[Document], max_cards: int = 3):
    """Display vendor information cards"""
    vendor_cards = []
    
    for doc in source_docs[:max_cards]:
        vendor_info = extract_vendor_info(doc.page_content)
        if vendor_info["name"]:
            vendor_cards.append(vendor_info)
    
    if vendor_cards:
        st.subheader("🔍 関連ベンダー情報")
        for i, vendor in enumerate(vendor_cards):
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background-color: #f9f9f9;">
                    <h4>📋 {vendor['name']}</h4>
                    <p><strong>強み:</strong> {vendor['strength']}</p>
                    <p><strong>URL:</strong> <a href="{vendor['url']}" target="_blank">{vendor['url']}</a></p>
                </div>
                """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AIベンダー検索（AWS Bedrock版）",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AIベンダー検索（AWS Bedrock版）")
    st.markdown("ベンダー調査データからAIベンダー情報を検索できます（AWS Bedrock使用）")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # Check vectorstore status
        vectorstore_exists = os.path.exists(VECTORSTORE_DIR) and any(Path(VECTORSTORE_DIR).iterdir())
        if vectorstore_exists:
            st.success("✅ インデックス: ロード済み")
        else:
            st.info("🔄 インデックス: 新規作成が必要")
        
        # Top K setting
        top_k = st.slider("検索件数 (Top K)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        
        # Re-index button
        if st.button("🔄 再インデックス", help="ベクトルストアを再作成します"):
            if os.path.exists(VECTORSTORE_DIR):
                import shutil
                shutil.rmtree(VECTORSTORE_DIR)
            st.rerun()
        
        # Debug info
        with st.expander("🐛 デバッグ情報"):
            st.write(f"ベクトルストア: {VECTORSTORE_DIR}")
            st.write(f"検索対象ファイル: {MD_PATHS}")
            st.write(f"チャンクサイズ: {CHUNK_SIZE}")
            st.write("LLM: anthropic.claude-v2")
            st.write("Embeddings: amazon.titan-embed-text-v1")
            st.write("リージョン: us-east-1")
    
    # Main content
    try:
        # Load documents
        docs = load_md(MD_PATHS)
        if not docs:
            st.stop()
        
        # Build or load vectorstore
        vectorstore = build_or_load_vectorstore(docs, VECTORSTORE_DIR)
        
        # Create chain
        chain = make_chain(vectorstore, top_k)
        
        # Search interface
        st.subheader("🔍 ベンダー検索")
        query = st.text_input(
            "質問を入力してください",
            placeholder="例: 法務系で安いベンダーは？",
            key="search_input"
        )
        
        if st.button("検索", type="primary") or query:
            if query:
                with st.spinner("検索中..."):
                    try:
                        result = chain({"query": query})
                        
                        # Display main answer
                        st.subheader("📝 回答")
                        st.write(result["result"])
                        
                        # Display vendor cards
                        if result.get("source_documents"):
                            display_vendor_cards(result["source_documents"])
                        
                        # Display source excerpts
                        if result.get("source_documents"):
                            with st.expander("📚 参照ソース抜粋", expanded=False):
                                for i, doc in enumerate(result["source_documents"][:3]):
                                    st.markdown(f"**ソース {i+1}:**")
                                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                    st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"検索エラー: {str(e)}")
                        st.info("AWS認証情報が正しく設定されているか確認してください（aws configure）")
            else:
                st.warning("質問を入力してください")
    
    except Exception as e:
        st.error(f"アプリケーションエラー: {str(e)}")
        st.info("AWS認証情報が正しく設定されているか確認してください（aws configure）")

if __name__ == "__main__":
    main()
#   U p d a t e d   0 9 / 2 1 / 2 0 2 5   1 1 : 5 0 : 4 5  
 