use rand::Rng;
use std::collections::VecDeque;

/// 離散分布を O(1) でサンプリングする Alias Table
pub struct AliasTable {
    pub prob: Vec<f32>,  // 各スロットのメイン確率 (0..=1)
    pub alias: Vec<u32>, // メインが外れたときの代替インデックス
    pub sum_w: f64,      // 元の重みの総和（pdf計算に使う）
}

impl AliasTable {
    /// 重み配列から AliasTable を構築する
    /// weights[i] >= 0 を想定
    pub fn new(weights: &[f32]) -> Self {
        let n = weights.len();
        assert!(n > 0);

        // 総和
        let sum_w: f64 = weights.iter().map(|&w| w.max(0.0) as f64).sum();

        // 全てゼロの場合は一様分布にする
        if sum_w == 0.0 {
            let prob = vec![1.0f32; n];
            let alias = (0..n as u32).collect();
            return AliasTable {
                prob,
                alias,
                sum_w: n as f64,
            };
        }

        // 正規化された確率を N 倍したもの
        // p[i] = weights[i] / sum_w * N
        let n_f = n as f64;
        let mut p: Vec<f64> = weights
            .iter()
            .map(|&w| (w.max(0.0) as f64) * n_f / sum_w)
            .collect();

        let mut small = VecDeque::new();
        let mut large = VecDeque::new();

        for (i, &pi) in p.iter().enumerate() {
            if pi < 1.0 {
                small.push_back(i);
            } else {
                large.push_back(i);
            }
        }

        let mut prob = vec![0.0f32; n];
        let mut alias = vec![0u32; n];

        // Vose のアルゴリズム
        while let (Some(s), Some(l)) = (small.pop_front(), large.pop_front()) {
            prob[s] = p[s] as f32;
            alias[s] = l as u32;

            p[l] = (p[l] + p[s]) - 1.0; // p[l] -= (1 - p[s])

            if p[l] < 1.0 {
                small.push_back(l);
            } else {
                large.push_back(l);
            }
        }

        // 残りは全て 1 に丸める
        for i in large.into_iter().chain(small.into_iter()) {
            prob[i] = 1.0;
            alias[i] = i as u32;
        }

        AliasTable { prob, alias, sum_w }
    }

    /// 1 サンプル取得（添字を返す）
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let n = self.prob.len();
        debug_assert!(n > 0);

        let i = rng.random_range(0..n);
        let r: f32 = rng.random(); // [0,1)
        if r < self.prob[i] {
            i
        } else {
            self.alias[i] as usize
        }
    }
}
