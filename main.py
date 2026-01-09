import os
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticQASystem:
    def __init__(self, config=None):
        self.config = {
            'document_root': 'data/A/A_document',
            'output_path': './result.csv',
            'test_path': 'data/A/A_question.csv',
            'model_name': 'BAAI/bge-small-zh',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 1,
            'drop_content': [
                '\x0c',
                '本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。'
            ]
        }

        if config:
            self.config.update(config)

        self.knowledge = ""
        self.chunks = []
        self.embeddings = []
        self.model = None

    def parse_pdf(self, pdf_path):

        try:
            flag = False
            if 'AZ' in pdf_path:
                flag = True

            text = ""
            with fitz.open(pdf_path) as doc:
                for n_page, page in enumerate(doc):
                    if flag and n_page < 2:
                        continue
                    text += page.get_text() + "\n"

            return text
        except Exception as e:
            logger.error(f"解析PDF文件 {pdf_path} 时出错: {str(e)}")
            return ""

    def drop_content_from_text(self, text):
        for content in self.config['drop_content']:
            text = text.replace(content, '')
        return text

    def split_text(self, text):

        chunks = []
        for i in range(0, len(text), self.config['chunk_size'] - self.config['chunk_overlap']):
            chunks.append(text[i:i + self.config['chunk_size']])
        return chunks

    def load_documents(self):
        logger.info("正在解析PDF文档...")
        documents = []

        try:
            for doc in tqdm(os.listdir(self.config['document_root'])):
                doc_path = os.path.join(self.config['document_root'], doc)
                text = self.parse_pdf(doc_path)
                if text:
                    text = self.drop_content_from_text(text)
                    documents.append(text)

            self.knowledge = ''.join(documents)
            logger.info(f"总文本长度: {len(self.knowledge)} 字符")

            logger.info("正在分块文本...")
            self.chunks = self.split_text(self.knowledge)
            logger.info(f"分块数量: {len(self.chunks)}")

            return True
        except Exception as e:
            logger.error(f"加载文档时出错: {str(e)}")
            return False

    def load_model(self):
        """加载语义模型"""
        try:
            logger.info(f"正在加载语义模型 {self.config['model_name']}...")
            self.model = SentenceTransformer(self.config['model_name'])
            return True
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return False

    def vectorize_chunks(self):
        if not self.model or not self.chunks:
            logger.error("模型或文本块未初始化")
            return False

        try:
            logger.info("正在生成文本向量...")
            # 批量处理以提高效率
            batch_size = 64
            self.embeddings = []

            for i in tqdm(range(0, len(self.chunks), batch_size)):
                batch = self.chunks[i:i + batch_size]
                batch_embeddings = self.model.encode(batch)
                self.embeddings.extend(batch_embeddings)

            return True
        except Exception as e:
            logger.error(f"向量化文本块时出错: {str(e)}")
            return False

    def semantic_search(self, query):
        if not self.model or not self.chunks or not self.embeddings:
            logger.error("系统未完全初始化")
            return None, 0

        try:
            query_embedding = self.model.encode(query)
            similarity_scores = [cosine_similarity([query_embedding], [emb])[0][0] for emb in self.embeddings]
            top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[
                :self.config['top_k']]

            results = [(self.chunks[i], similarity_scores[i]) for i in top_indices]
            return results[0] if results else (None, 0)
        except Exception as e:
            logger.error(f"语义搜索时出错: {str(e)}")
            return None, 0

    def process_questions(self):
        try:
            logger.info("正在处理问题...")
            df_test = pd.read_csv(self.config['test_path'])

            results = df_test['question'].apply(lambda x: self.semantic_search(x))
            df_test['answer'] = [r[0] for r in results]
            df_test['similarity'] = [r[1] for r in results]

            logger.info("正在保存结果...")
            df_test.to_csv(self.config['output_path'], index=False)
            logger.info(f"结果已保存至: {self.config['output_path']}")

            return True
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            return False

    def run(self):
        if not self.load_documents():
            return False

        if not self.load_model():
            return False

        if not self.vectorize_chunks():
            return False

        return self.process_questions()


if __name__ == "__main__":
    config = {
        'document_root': r'data/A/A_document',
        'output_path': r'./result.csv',
        'test_path': r'data/A/A_question.csv',
        'model_name': 'BAAI/bge-small-zh',
        'chunk_size': 500,
        'chunk_overlap': 50,
        'top_k': 1
    }

    qa_system = SemanticQASystem(config)
    qa_system.run()