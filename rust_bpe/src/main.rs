use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::fs;

fn main() {
    // 读取语料库文件
    let corpus_path = "../tests/siku.txt";
    let corpus_content = match fs::read_to_string(corpus_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("无法读取文件 {}: {}", corpus_path, e);
            return;
        }
    };

    // 将语料库分割为行
    let texts: Vec<String> = corpus_content.lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    println!("加载语料库完成，共 {} 行文本", texts.len());

    // 特殊标记
    let special_tokens = vec!["<unk>".to_string(), "<pad>".to_string()];

    // 设置词表大小
    let vocab_size = 512;

    // 计时开始
    let start_time = Instant::now();

    // 训练BPE
    let bpe = BPE::train(&texts, vocab_size, &special_tokens);

    // 计时结束
    let duration = start_time.elapsed();
    println!("\n训练完成，耗时: {:.2}秒", duration.as_secs_f32());

    println!("\n词汇表大小: {}", bpe.vocab.len());
    println!("合并规则数量: {}", bpe.merges.len());
}

pub struct BPE {
    pub vocab: HashSet<String>,
    pub merges: Vec<(String, String)>,
    pub special_tokens: HashMap<String, usize>,
}

impl BPE {
    pub fn train(
        texts: &[String],
        vocab_size: usize,
        special_tokens: &[String],
    ) -> Self {
        // 初始化词汇表为256个字节 (0x00 到 0xFF)
        let mut vocab: HashSet<String> = (0..=255u8)
            .map(|b| format!("{:02x}", b))
            .collect();

        // 初始合并规则为空
        let mut merges = Vec::new();

        // 将文本转换为字节序列的十六进制表示
        let tokenized_texts: Vec<Vec<String>> = texts
            .iter()
            .map(|text| {
                text.as_bytes()
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect()
            })
            .collect();

        // 使用互斥锁保护共享状态
        let tokenized_texts = Arc::new(Mutex::new(tokenized_texts));

        // 计算需要合并的次数（不再预留特殊标记空间）
        let total_merges = vocab_size.saturating_sub(vocab.len());
        println!("初始词汇表大小: {}", vocab.len());
        println!("计划合并次数: {}", total_merges);

        // BPE训练循环
        let mut merge_count = 0;
        while vocab.len() < vocab_size && merge_count < total_merges {
            // 使用线程池统计所有相邻token对的出现频率
            let pair_counts = Arc::new(Mutex::new(HashMap::new()));
            let mut handles = vec![];

            // 获取锁并克隆数据用于线程
            let texts_clone = tokenized_texts.lock().unwrap().clone();
            let num_chunks = num_cpus::get().max(1);
            let chunk_size = (texts_clone.len() + num_chunks - 1) / num_chunks;

            for chunk in texts_clone.chunks(chunk_size) {
                let chunk = chunk.to_vec();
                let pair_counts_clone = Arc::clone(&pair_counts);

                handles.push(thread::spawn(move || {
                    let mut local_counts = HashMap::new();

                    for tokens in &chunk {
                        for pair in tokens.windows(2) {
                            if pair[0].is_empty() || pair[1].is_empty() {
                                continue;
                            }
                            *local_counts.entry((pair[0].clone(), pair[1].clone())).or_insert(0) += 1;
                        }
                    }

                    let mut global_counts = pair_counts_clone.lock().unwrap();
                    for (pair, count) in local_counts {
                        *global_counts.entry(pair).or_insert(0) += count;
                    }
                }));
            }

            // 等待所有线程完成
            for handle in handles {
                handle.join().unwrap();
            }

            let pair_counts = Arc::try_unwrap(pair_counts).unwrap().into_inner().unwrap();

            // 如果没有找到有效的token对，提前终止
            if pair_counts.is_empty() {
                println!("没有更多可合并的token对，提前终止训练");
                break;
            }

            // 串行决策阶段：选择频率最高的token对
            let (best_pair, best_count) = pair_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, count)| (pair.clone(), *count))
                .unwrap();

            // 创建新token
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            vocab.insert(new_token.clone());
            merges.push(best_pair.clone());

            // 打印合并信息
            merge_count += 1;
            println!("merge {}/{}: ({}, {}) -> {} had {} occurrences",
                     merge_count, total_merges,
                     best_pair.0, best_pair.1,
                     new_token, best_count);

            // 更新token序列：合并选中的token对
            let mut new_tokenized = Vec::new();
            for tokens in tokenized_texts.lock().unwrap().iter() {
                let mut new_tokens = Vec::new();
                let mut i = 0;

                while i < tokens.len() {
                    if i < tokens.len() - 1
                        && tokens[i] == best_pair.0
                        && tokens[i + 1] == best_pair.1 {
                        new_tokens.push(new_token.clone());
                        i += 2; // 跳过两个token
                    } else {
                        new_tokens.push(tokens[i].clone());
                        i += 1;
                    }
                }
                new_tokenized.push(new_tokens);
            }

            *tokenized_texts.lock().unwrap() = new_tokenized;
        }

        // 添加特殊标记到词表
        for token in special_tokens {
            vocab.insert(token.clone());
        }

        // 构建特殊标记映射
        let special_tokens_map: HashMap<String, usize> = special_tokens
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), vocab_size - special_tokens.len() + i))
            .collect();

        BPE {
            vocab,
            merges,
            special_tokens: special_tokens_map,
        }
    }

    // 编码函数（优先处理特殊标记）
    pub fn encode(&self, text: &str) -> Vec<String> {
        // 检查特殊标记
        for token in self.special_tokens.keys() {
            if text.contains(token) {
                // 简化处理：如果包含特殊标记，直接返回整个文本作为token
                // 实际应用中需要更精细的处理（如分词）
                return vec![text.to_string()];
            }
        }

        // 将文本转换为字节序列的十六进制表示
        let bytes = text.as_bytes();
        let mut tokens: Vec<String> = bytes.iter().map(|b| format!("{:02x}", b)).collect();

        // 应用合并规则
        for (left, right) in &self.merges {
            let merged = format!("{}{}", left, right);
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == *left && tokens[i + 1] == *right {
                    tokens.splice(i..=i + 1, [merged.clone()]);
                    // 继续检查当前位置（新token可能与后续token形成新的组合）
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }
}
