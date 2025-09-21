import re
import time
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import ttk, scrolledtext
import random
import json
import math
import multiprocessing
from functools import partial

# 基础分词器类
class BaseTokenizer:
    def __init__(self, special_tokens=None):
        self.vocab = {}
        self.special_tokens = special_tokens or []
        self.pattern = r"""\w+|\S"""
        self.compiled_pattern = re.compile(self.pattern)
        self.token_frequencies = Counter()
        self.name = "Base"
        self.training_time = 0.0  # 训练时间记录

    def _split_words(self, text):
        """使用正则表达式分割文本为单词"""
        return self.compiled_pattern.findall(text)

    def _compute_token_frequencies(self, corpus):
        """计算语料库中每个token的频率"""
        self.token_frequencies = Counter()
        for text in corpus:
            tokens = self.tokenize(text)
            self.token_frequencies.update(tokens)

    def train(self, corpus, vocab_size, num_cores=1):
        raise NotImplementedError("Subclasses must implement this method")

    def tokenize(self, text):
        raise NotImplementedError("Subclasses must implement this method")

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get('<unk>', -1)) for token in tokens]

    def decode(self, token_ids):
        tokens = []
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        for token_id in token_ids:
            token = id_to_token.get(token_id)
            if token is not None:
                tokens.append(token)
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    def get_token_colors(self):
        token_colors = {}
        base_colors = [
            "#FF9999", "#99FF99", "#9999FF", "#FFFF99", "#FF99FF", "#99FFFF",
            "#FFCC99", "#CCFF99", "#99CCFF", "#FF99CC", "#CC99FF", "#99FFCC",
            "#FFB366", "#B3FF66", "#66B3FF", "#FF66B3", "#B366FF", "#66FFB3",
            "#FF8080", "#80FF80", "#8080FF", "#FFFF80", "#FF80FF", "#80FFFF"
        ]

        color_index = 0
        for token in self.vocab.keys():
            if token not in token_colors:
                token_colors[token] = base_colors[color_index % len(base_colors)]
                color_index += 1

        return token_colors

# BPE分词器（增加并行化）
class BPETokenizer(BaseTokenizer):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens)
        self.merges = []
        self.name = "BPE"

    def _count_pairs(self, chunk):
        """计算一个语料块中的符号对频率"""
        pair_counts = defaultdict(int)
        for token, freq in chunk:  # 修改这里：解包元组
            symbols = token.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_counts[pair] += freq  # 使用频率计数
        return pair_counts

    def train(self, corpus, vocab_size, num_cores=1):
        start_time = time.time()  # 记录开始时间
        
        # 初始化词汇表
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(self.vocab)

        # 统计词频
        word_freq = Counter()
        for text in corpus:
            words = self._split_words(text)
            word_freq.update(words)

        # 初始化基础词汇（字符级）
        vocab = defaultdict(int)
        for word, freq in word_freq.items():
            if word in self.special_tokens:
                continue

            for char in word:
                vocab[char] += freq
            vocab[' '.join(list(word)) + ' </w>'] = freq

        vocab = {k: v for k, v in vocab.items()}

        # BPE 合并迭代
        while len(vocab) < vocab_size:
            # 并行统计符号对频率
            vocab_items = list(vocab.items())
            
            if num_cores > 1:
                chunk_size = max(1, len(vocab_items) // num_cores)
                chunks = [vocab_items[i:i+chunk_size] for i in range(0, len(vocab_items), chunk_size)]
                
                with multiprocessing.Pool(num_cores) as pool:
                    results = pool.map(self._count_pairs, chunks)
                
                # 合并结果
                pairs = defaultdict(int)
                for result in results:
                    for pair, count in result.items():
                        pairs[pair] += count
            else:
                # 单核处理
                pairs = defaultdict(int)
                for token, freq in vocab.items():
                    symbols = token.split()
                    for i in range(len(symbols) - 1):
                        pair = (symbols[i], symbols[i+1])
                        pairs[pair] += freq

            if not pairs:
                break

            # 选择最高频的符号对
            best_pair = max(pairs, key=pairs.get)
            new_symbol = ''.join(best_pair)

            # 记录合并操作
            self.merges.append(best_pair)

            # 更新词汇表
            new_vocab = {}
            for token, freq in vocab.items():
                new_token = token.replace(' '.join(best_pair), new_symbol)
                new_vocab[new_token] = freq

            vocab = new_vocab

        # 构建最终词汇表
        for token in vocab:
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

        # 统计token频率
        self._compute_token_frequencies(corpus)
        
        end_time = time.time()  # 记录结束时间
        self.training_time = end_time - start_time  # 计算训练时间

    def tokenize(self, text):
        tokens = []
        for word in self._split_words(text):
            if word in self.special_tokens:
                tokens.append(word)
                continue

            sequence = list(word) + ['</w>']

            for pair in self.merges:
                new_sequence = []
                i = 0
                while i < len(sequence):
                    if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i+1] == pair[1]:
                        new_sequence.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_sequence.append(sequence[i])
                        i += 1
                sequence = new_sequence

            tokens.extend(sequence)

        return tokens

# WordPiece分词器（简化实现）
class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens)
        self.name = "WordPiece"

    def train(self, corpus, vocab_size, num_cores=1):
        start_time = time.time()  # 记录开始时间
        
        # 简化实现：类似于BPE但使用不同的评分函数
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(self.vocab)

        word_freq = Counter()
        for text in corpus:
            words = self._split_words(text)
            word_freq.update(words)

        # 初始化字符级词汇表
        vocab = defaultdict(int)
        for word, freq in word_freq.items():
            if word in self.special_tokens:
                continue

            for char in word:
                vocab[char] += freq
            vocab[' '.join(list(word)) + ' </w>'] = freq

        vocab = {k: v for k, v in vocab.items()}

        # WordPiece风格的合并（使用不同的评分）
        while len(vocab) < vocab_size:
            pairs = defaultdict(int)
            for token, freq in vocab.items():
                symbols = token.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i+1])
                    # WordPiece使用不同的评分：freq(pair) / (freq(first) * freq(second))
                    score = freq / (vocab.get(symbols[i], 1) * vocab.get(symbols[i+1], 1))
                    pairs[pair] += score

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_symbol = ''.join(best_pair)

            # 更新词汇表
            new_vocab = {}
            for token, freq in vocab.items():
                new_token = token.replace(' '.join(best_pair), new_symbol)
                new_vocab[new_token] = freq

            vocab = new_vocab

        for token in vocab:
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

        self._compute_token_frequencies(corpus)
        
        end_time = time.time()  # 记录结束时间
        self.training_time = end_time - start_time  # 计算训练时间

    def tokenize(self, text):
        # 简化实现：使用类似BPE的方法
        tokens = []
        for word in self._split_words(text):
            if word in self.special_tokens:
                tokens.append(word)
                continue

            # 这里使用一个简化的WordPiece分词
            # 实际WordPiece会使用最大前向匹配
            tokens.extend(list(word))
            tokens.append('</w>')

        return tokens

# Unigram分词器（简化实现）
class UnigramTokenizer(BaseTokenizer):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens)
        self.name = "Unigram"
        self.token_scores = {}

    def train(self, corpus, vocab_size, num_cores=1):
        start_time = time.time()  # 记录开始时间
        
        # 简化实现：初始化一个大的词汇表然后修剪
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(self.vocab)

        # 初始词汇表包含所有字符和常见子串
        char_vocab = set()
        for text in corpus:
            words = self._split_words(text)
            for word in words:
                if word in self.special_tokens:
                    continue
                char_vocab.update(list(word))
                # 添加一些常见子串
                if len(word) > 1:
                    for i in range(len(word) - 1):
                        char_vocab.add(word[i:i+2])

        # 添加到词汇表
        for token in char_vocab:
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

        # 简化：随机分配分数
        for token in self.vocab:
            if token not in self.special_tokens:
                self.token_scores[token] = random.random()

        self._compute_token_frequencies(corpus)
        
        end_time = time.time()  # 记录结束时间
        self.training_time = end_time - start_time  # 计算训练时间

    def tokenize(self, text):
        # 简化实现：使用贪心算法选择最高分的分词
        tokens = []
        for word in self._split_words(text):
            if word in self.special_tokens:
                tokens.append(word)
                continue

            # 简化的Unigram分词：按字符分割
            tokens.extend(list(word))
            tokens.append('</w>')

        return tokens

# SentencePiece分词器（增加并行化）
class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens)
        self.name = "SentencePiece"
        self.merges = []  # 存储合并规则
        self.token_frequencies = Counter()

    def _count_pairs(self, sequences_chunk):
        """计算一个序列块中的符号对频率"""
        pair_counts = defaultdict(int)
        for seq in sequences_chunk:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += 1
        return pair_counts

    def _apply_merge(self, sequences_chunk, best_pair, new_symbol):
        """在序列块中应用合并操作"""
        new_sequences = []
        for seq in sequences_chunk:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                    new_seq.append(new_symbol)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        return new_sequences

    def train(self, corpus, vocab_size, num_cores=1):
        start_time = time.time()  # 记录开始时间
        
        # 初始化特殊token
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(self.vocab)

        # 1. 构建初始字符级词汇表
        char_freq = Counter()
        for text in corpus:
            for char in text:
                char_freq[char] += 1
            char_freq['</w>'] += 1  # 添加词尾标记

        # 添加字符到词汇表
        for char in char_freq:
            if char not in self.vocab:
                self.vocab[char] = next_id
                next_id += 1

        # 2. 准备初始token序列（字符级）
        sequences = []
        for text in corpus:
            # 字符列表 + 结束标记
            seq = list(text) + ['</w>']
            sequences.append(seq)

        # 3. BPE风格的合并迭代
        while len(self.vocab) < vocab_size:
            # 统计相邻符号对频率
            if num_cores > 1:
                # 并行统计
                chunk_size = max(1, len(sequences) // num_cores)
                chunks = [sequences[i:i+chunk_size] for i in range(0, len(sequences), chunk_size)]
                
                with multiprocessing.Pool(num_cores) as pool:
                    results = pool.map(self._count_pairs, chunks)
                
                # 合并结果
                pair_counts = defaultdict(int)
                for result in results:
                    for pair, count in result.items():
                        pair_counts[pair] += count
            else:
                # 单核处理
                pair_counts = defaultdict(int)
                for seq in sequences:
                    for i in range(len(seq) - 1):
                        pair = (seq[i], seq[i+1])
                        pair_counts[pair] += 1

            if not pair_counts:
                break

            # 选择最高频的符号对
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = ''.join(best_pair)

            # 记录合并规则
            self.merges.append(best_pair)

            # 添加到词汇表
            if new_token not in self.vocab:
                self.vocab[new_token] = next_id
                next_id += 1

            # 更新所有序列（原地合并）
            if num_cores > 1:
                # 并行更新序列
                apply_merge_partial = partial(self._apply_merge, best_pair=best_pair, new_symbol=new_token)
                with multiprocessing.Pool(num_cores) as pool:
                    new_sequences_chunks = pool.map(apply_merge_partial, chunks)
                
                # 合并更新后的序列
                sequences = []
                for chunk in new_sequences_chunks:
                    sequences.extend(chunk)
            else:
                # 单核更新序列
                new_sequences = []
                for seq in sequences:
                    new_seq = []
                    i = 0
                    while i < len(seq):
                        if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                            new_seq.append(new_token)
                            i += 2
                        else:
                            new_seq.append(seq[i])
                            i += 1
                    new_sequences.append(new_seq)
                sequences = new_sequences

        # 4. 计算最终token频率
        self._compute_token_frequencies(corpus)
        
        end_time = time.time()  # 记录结束时间
        self.training_time = end_time - start_time  # 计算训练时间

    def tokenize(self, text):
        # 初始化为字符序列
        tokens = list(text) + ['</w>']

        # 应用所有合并规则
        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_pair:
                    new_tokens.append(''.join(merge_pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

# Byte-level BPE分词器（简化实现）
class ByteLevelBPETokenizer(BaseTokenizer):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens)
        self.merges = []
        self.name = "Byte-level BPE"

    def train(self, corpus, vocab_size, num_cores=1):
        start_time = time.time()  # 记录开始时间
        
        # 简化实现：在字节级别上运行BPE
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(self.vocab)

        # 将文本转换为字节
        byte_freq = Counter()
        for text in corpus:
            byte_text = text.encode('utf-8')
            for byte in byte_text:
                byte_freq[byte] += 1
            # 添加字节序列
            byte_seq = ' '.join([str(b) for b in byte_text]) + ' </w>'
            byte_freq[byte_seq] += 1

        # 初始化字节级词汇表
        vocab = defaultdict(int)
        for token, freq in byte_freq.items():
            if isinstance(token, int):  # 单个字节
                vocab[str(token)] = freq
            else:  # 字节序列
                vocab[token] = freq

        vocab = {k: v for k, v in vocab.items()}

        # BPE合并
        while len(vocab) < vocab_size:
            pairs = defaultdict(int)
            for token, freq in vocab.items():
                symbols = token.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_symbol = ''.join(best_pair)

            new_vocab = {}
            for token, freq in vocab.items():
                new_token = token.replace(' '.join(best_pair), new_symbol)
                new_vocab[new_token] = freq

            vocab = new_vocab

        for token in vocab:
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

        self._compute_token_frequencies(corpus)
        
        end_time = time.time()  # 记录结束时间
        self.training_time = end_time - start_time  # 计算训练时间

    def tokenize(self, text):
        # 简化实现：将文本转换为字节表示
        tokens = []
        byte_text = text.encode('utf-8')
        for byte in byte_text:
            tokens.append(str(byte))
        tokens.append('</w>')
        return tokens

# 分词器工厂
class TokenizerFactory:
    @staticmethod
    def create_tokenizer(tokenizer_type, special_tokens=None):
        if tokenizer_type == "BPE":
            return BPETokenizer(special_tokens)
        elif tokenizer_type == "WordPiece":
            return WordPieceTokenizer(special_tokens)
        elif tokenizer_type == "Unigram":
            return UnigramTokenizer(special_tokens)
        elif tokenizer_type == "SentencePiece":
            return SentencePieceTokenizer(special_tokens)
        elif tokenizer_type == "Byte-level BPE":
            return ByteLevelBPETokenizer(special_tokens)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

class TokenizerVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("多分词器可视化工具")
        self.root.geometry("1200x900")

        # 支持并行化的分词器类型
        self.parallel_tokenizers = ["BPE", "SentencePiece"]
        # 全部分词器类型
        self.all_tokenizers = ["BPE", "WordPiece", "Unigram", "SentencePiece", "Byte-level BPE"]
        
        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 顶部控制区域
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="选择分词器:").grid(row=0, column=0, sticky=tk.W)
        self.tokenizer_var = tk.StringVar()
        # 初始显示全部分词器
        self.tokenizer_combo = ttk.Combobox(control_frame, textvariable=self.tokenizer_var,
                                           values=self.all_tokenizers, state="readonly")
        self.tokenizer_combo.set("BPE")  # 默认选择BPE
        self.tokenizer_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.tokenizer_combo.bind("<<ComboboxSelected>>", self.on_tokenizer_change)

        ttk.Label(control_frame, text="词汇表大小:").grid(row=0, column=2, sticky=tk.W)
        self.vocab_size = ttk.Entry(control_frame, width=10)
        self.vocab_size.insert(0, "150")
        self.vocab_size.grid(row=0, column=3, sticky=tk.W, padx=5)

        # 并行化选项
        self.parallel_var = tk.BooleanVar(value=False)
        self.parallel_check = ttk.Checkbutton(control_frame, text="并行化", variable=self.parallel_var,
                                             command=self.toggle_parallel_options)
        self.parallel_check.grid(row=0, column=4, sticky=tk.W, padx=5)
        
        ttk.Label(control_frame, text="核数:").grid(row=0, column=5, sticky=tk.W)
        self.num_cores = tk.Spinbox(control_frame, from_=2, to=16, width=5)
        self.num_cores.delete(0, tk.END)
        self.num_cores.insert(0, "4")
        self.num_cores.config(state='disabled')  # 初始禁用
        self.num_cores.grid(row=0, column=6, sticky=tk.W, padx=5)

        self.train_btn = ttk.Button(control_frame, text="训练分词器", command=self.train_tokenizer)
        self.train_btn.grid(row=0, column=7, sticky=tk.W, padx=5)

        # 左右对比区域
        comparison_frame = ttk.Frame(self.root, padding="10")
        comparison_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.N, tk.S, tk.W, tk.E))

        # 左侧：语料库输入
        left_frame = ttk.LabelFrame(comparison_frame, text="语料库输入", padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5, pady=5)
        self.corpus_text = scrolledtext.ScrolledText(left_frame, width=60, height=20)
        self.corpus_text.pack(fill=tk.BOTH, expand=True)

        # 右侧：分词结果
        right_frame = ttk.LabelFrame(comparison_frame, text="分词结果", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5, pady=5)
        self.tokenized_text = scrolledtext.ScrolledText(right_frame, width=60, height=20)
        self.tokenized_text.pack(fill=tk.BOTH, expand=True)

        # 配置对比区域的权重
        comparison_frame.columnconfigure(0, weight=1)
        comparison_frame.columnconfigure(1, weight=1)
        comparison_frame.rowconfigure(0, weight=1)

        # 底部标签页区域
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.N, tk.S, tk.W, tk.E), padx=10, pady=10)

        # Token统计标签
        self.stats_frame = ttk.Frame(notebook)
        notebook.add(self.stats_frame, text="Token统计")
        self.stats_text = scrolledtext.ScrolledText(self.stats_frame, width=100, height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Tokenization结果标签
        self.encoded_frame = ttk.Frame(notebook)
        notebook.add(self.encoded_frame, text="Tokenization结果")
        self.encoded_text = scrolledtext.ScrolledText(self.encoded_frame, width=100, height=15)
        self.encoded_text.pack(fill=tk.BOTH, expand=True)

        # 配置权重使组件可以扩展
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)  # 对比区域
        self.root.rowconfigure(2, weight=1)  # 标签页区域
        control_frame.columnconfigure(0, weight=1)

        # 初始化分词器
        self.on_tokenizer_change()

    def toggle_parallel_options(self):
        """切换并行化选项的状态"""
        if self.parallel_var.get():
            # 启用核数输入框
            self.num_cores.config(state='normal')
            # 更新分词器列表为支持并行化的算法
            self.tokenizer_combo.config(values=self.parallel_tokenizers)
            # 确保当前选择的分词器在列表中
            if self.tokenizer_var.get() not in self.parallel_tokenizers:
                self.tokenizer_var.set(self.parallel_tokenizers[0])
        else:
            # 禁用核数输入框
            self.num_cores.config(state='disabled')
            # 恢复全部分词器列表
            self.tokenizer_combo.config(values=self.all_tokenizers)
            # 确保当前选择的分词器在列表中
            if self.tokenizer_var.get() not in self.all_tokenizers:
                self.tokenizer_var.set(self.all_tokenizers[0])

    def on_tokenizer_change(self, event=None):
        tokenizer_type = self.tokenizer_var.get()
        self.current_tokenizer = TokenizerFactory.create_tokenizer(
            tokenizer_type, special_tokens=['<unk>', '<pad>'])

    def train_tokenizer(self):
        if self.current_tokenizer is None:
            return

        # 获取语料库
        corpus_text = self.corpus_text.get("1.0", tk.END).strip()
        if not corpus_text:
            return

        corpus = [line for line in corpus_text.split('\n') if line.strip()]

        # 获取词汇表大小
        try:
            vocab_size = int(self.vocab_size.get())
        except ValueError:
            return

        # 获取并行化设置
        use_parallel = self.parallel_var.get()
        try:
            num_cores = int(self.num_cores.get()) if use_parallel else 1
        except ValueError:
            num_cores = 4 if use_parallel else 1

        # 训练分词器
        self.current_tokenizer.train(corpus, vocab_size, num_cores)

        # 更新显示
        self.update_displays()

    def update_displays(self):
        # 显示分词结果（带颜色）
        self.tokenized_text.delete("1.0", tk.END)
        token_colors = self.current_tokenizer.get_token_colors()

        # 配置文本样式
        self.tokenized_text.configure(font=("Courier", 10))

        # 获取语料库
        corpus_text = self.corpus_text.get("1.0", tk.END).strip()
        corpus = [line for line in corpus_text.split('\n') if line.strip()]

        # 在分词结果顶部显示当前分词器类型和并行化信息
        use_parallel = self.parallel_var.get()
        num_cores = self.num_cores.get() if use_parallel else "1"
        parallel_info = f"并行化: {'是' if use_parallel else '否'}, 核数: {num_cores}" if use_parallel else "并行化: 否"
        
        self.tokenized_text.insert(tk.END, f"当前分词器: {self.current_tokenizer.name} ({parallel_info})\n\n")

        for text in corpus:
            tokens = self.current_tokenizer.tokenize(text)
            for token in tokens:
                color = token_colors.get(token, "#FFFFFF")
                tag_name = f"token_{hash(token)}"

                # 配置标签样式
                self.tokenized_text.tag_configure(tag_name, background=color,
                                                 foreground="black",
                                                 borderwidth=1,
                                                 relief="solid",
                                                 font=("Courier", 10, "bold"))

                # 插入带样式的token
                self.tokenized_text.insert(tk.END, token + " ", tag_name)
            self.tokenized_text.insert(tk.END, "\n")

        # 显示token统计
        self.stats_text.delete("1.0", tk.END)
        total_tokens = sum(self.current_tokenizer.token_frequencies.values())
        unique_tokens = len(self.current_tokenizer.token_frequencies)

        # 获取并行化设置
        use_parallel = self.parallel_var.get()
        num_cores = self.num_cores.get() if use_parallel else "1"
        parallel_info = f"并行化: {'是' if use_parallel else '否'}, 核数: {num_cores}" if use_parallel else "并行化: 否"

        # 格式化训练时间显示
        training_time = self.current_tokenizer.training_time
        time_str = f"{training_time:.4f} 秒"
        if training_time > 1.0:
            time_str = f"{training_time:.2f} 秒"
        if training_time > 60.0:
            minutes = int(training_time // 60)
            seconds = training_time % 60
            time_str = f"{minutes}分{seconds:.2f}秒"

        self.stats_text.insert(tk.END, f"分词器类型: {self.current_tokenizer.name}\n")
        self.stats_text.insert(tk.END, f"训练配置: {parallel_info}\n")
        self.stats_text.insert(tk.END, f"训练时间: {time_str}\n")
        self.stats_text.insert(tk.END, f"总Token数量: {total_tokens}\n")
        self.stats_text.insert(tk.END, f"唯一Token数量: {unique_tokens}\n\n")
        self.stats_text.insert(tk.END, "Token频率统计:\n")

        for token, freq in self.current_tokenizer.token_frequencies.most_common():
            self.stats_text.insert(tk.END, f"'{token}': {freq}\n")

        # 显示tokenization结果
        self.encoded_text.delete("1.0", tk.END)
        for text in corpus:
            token_ids = self.current_tokenizer.encode(text)
            self.encoded_text.insert(tk.END, f"文本: {text}\n")
            self.encoded_text.insert(tk.END, f"Token IDs: {token_ids}\n\n")

# 示例语料库
sample_corpus = [
    "Tokenization is at the heart of LLMs. Do not brush it off.",
    "",
    "127 + 677 = 804",
    "1275 + 6773 = 8041",
    "",
    "Apple.",
    "I have an Apple.",
    "apple.",
    "APPLE.",
    "",
    "很高兴在ECNU DASE见到你。这里是胡仁君老师的《从零开始语言模型构建实践》。如果你有任何疑问，请随时问我。",
    "",
    "ECNU DASE에서 만나서 반갑습니다. 여기는 호인군 선생님의 '처음부터 시작하는 언어 모델 구축 실습' 시간입니다. 질문이 있으시면 언제든지 물어보세요.",
    "",
    "Es freut mich, Sie an der ECNU DASE zu treffen. Dies ist Hu Renjuns Kurs 'Praktische Übungen zum Aufbau von Sprachmodellen von Grund auf'. Wenn Sie Fragen haben, stehe ich Ihnen gerne zur Verfügung.",
    "",
    "for i in range(1, 101):",
    "    if i % 3 == 0 and i % 5 == 0:",
    "        print(\"FizzBuzz\")",
    "    elif i % 3 == 0:",
    "        print(\"Fizz\")",
    "    elif i % 5 == 0:",
    "        print(\"Buzz\")",
    "    else:",
    "        print(i)",
    "",
    "Let's enjoy tokenizer and bid on it NOW!",
    "* BPE (Byte Pair Encoding)：基于频率的合并算法",
    "* WordPiece：类似于BPE但使用不同的评分函数",
    "* Unigram：基于概率的语言模型分词",
    "* SentencePiece：直接在原始文本上操作的分词器",
    "* Byte-level BPE：在字节级别运行的BPE变体"
]

def main():
    root = tk.Tk()
    app = TokenizerVisualizer(root)

    # 预填充示例语料库
    app.corpus_text.insert("1.0", "\n".join(sample_corpus))

    root.mainloop()

if __name__ == "__main__":
    main()
