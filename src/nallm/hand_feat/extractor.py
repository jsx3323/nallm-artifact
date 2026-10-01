"""手工特征提取器

实现 36 维手工特征的计算逻辑。
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyjarowinkler import distance
from unidecode import unidecode

from nallm.hand_feat.config import IDF_DIR, DEFAULT_IDF_VALUES
from nallm.hand_feat.name_match import MatchName

logger = logging.getLogger(__name__)

# 化学式/数字噪声清洗(7.2 根因③:碎片 token 占 top 候选 score_author 贡献 27%)
_TITLE_NUM = re.compile(r"\d+(?:\.\d+)?")


def normalize_title(text: str) -> str:
    """title 数字噪声清洗:剥离 token 内配比/版本数字,删孤立数字 token

    La0.7Mg0.3 → LaMg, Fe2O3 → FeO, 孤立数字(0/55) → 删除。
    不同配比的同系化合物清洗后可正确匹配,年份/版本号视为噪声剔除。
    """
    return " ".join(_TITLE_NUM.sub("", text).split())


class HandFeatureExtractor:
    """36 维手工特征提取器"""

    def __init__(self):
        self._load_essentials()

    def _load_essentials(self):
        """加载 IDF 字典等必要数据"""
        # 加载 JSON 文件
        files_to_load = [
            ("name_uniq_dict.json", "name_uniq_dict"),
            ("venue_idf.json", "ven_tfidf"),
            ("new_org_idf.json", "org_tfidf"),
            ("title_idf.json", "title_tfidf"),
        ]

        for filename, attr_name in files_to_load:
            file_path = IDF_DIR / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    setattr(self, attr_name, json.load(f))
            else:
                logger.warning(f"IDF 文件不存在: {file_path}")
                setattr(self, attr_name, {})

        self.STOPWORDS = {"jr", "iii", "dr", "mr", "junior"}

        # 昵称字典
        self.NICKNAME_DICT = {
            "al": "albert",
            "andy": "andrew",
            "tony": "anthony",
            "art": "arthur",
            "arty": "arthur",
            "bernie": "bernard",
            "bern": "bernard",
            "charlie": "charles",
            "chuck": "charles",
            "danny": "daniel",
            "dan": "daniel",
            "don": "donald",
            "ed": "edward",
            "eddie": "edward",
            "gene": "eugene",
            "fran": "francis",
            "freddy": "frederick",
            "fred": "frederick",
            "hank": "henry",
            "irv": "irving",
            "jimmy": "james",
            "jim": "james",
            "joe": "joseph",
            "jacky": "john",
            "jack": "john",
            "jeff": "jeffrey",
            "ken": "kenneth",
            "larry": "lawrence",
            "leo": "leonard",
            "matt": "matthew",
            "mike": "michael",
            "nate": "nathan",
            "nat": "nathan",
            "nick": "nicholas",
            "pat": "patrick",
            "pete": "peter",
            "ray": "raymond",
            "dick": "richard",
            "rick": "richard",
            "bob": "robert",
            "bobby": "robert",
            "rob": "robert",
            "ron": "ronald",
            "ronny": "ronald",
            "russ": "russell",
            "sam": "samuel",
            "sammy": "samuel",
            "steve": "stephan",
            "stu": "stuart",
            "teddy": "theodore",
            "ted": "theodore",
            "tom": "thomas",
            "thom": "thomas",
            "tommy": "thomas",
            "timmy": "timothy",
            "tim": "timothy",
            "walt": "walter",
            "wally": "walter",
            "bill": "william",
            "billy": "william",
            "will": "william",
            "willy": "william",
            "mandy": "amanda",
            "cathy": "catherine",
            "cath": "catherine",
            "chris": "christopher",
            "chrissy": "christine",
            "cindy": "cynthia",
            "cynth": "cynthia",
            "debbie": "deborah",
            "deb": "deborah",
            "betty": "elizabeth",
            "beth": "elizabeth",
            "liz": "elizabeth",
            "bess": "elizabeth",
            "flo": "florence",
            "francie": "frances",
            "jan": "janet",
            "kate": "katherine",
            "kathy": "katherine",
            "janice": "janice",
            "nan": "nancy",
            "pam": "pamela",
            "bobbie": "roberta",
            "sophie": "sophia",
            "sue": "susan",
            "suzie": "susan",
            "terry": "teresa",
            "val": "valerie",
            "ronnie": "veronica",
            "vonna": "yvonne",
            "peggy": "margaret",
            "sally": "sarah",
            "harry": "henry",
        }

    def tokenize_name(self, name: str) -> str:
        """分词并标准化姓名"""
        splitted_name = []
        for word in name.split():
            if len(word) == 2 and word.count(".") == 0 and word.isupper():
                word = " ".join(word)
            splitted_name.append(word)
        name = " ".join(splitted_name).replace("'", "").replace("'", "")
        name = re.sub(r"[^\w.]", " ", name).lower()
        name = unidecode(name)
        splitted_name = []
        for word in name.split():
            if word.replace(".", "") in self.STOPWORDS:
                continue
            if word in self.NICKNAME_DICT:
                word = self.NICKNAME_DICT[word]
            if word.count(".") > 1:
                word = " ".join(word.split("."))
            splitted_name.append(word)
        name = " ".join(splitted_name)
        name = re.sub(r" +", " ", name.encode("ascii", "ignore").decode("ascii"))
        return name

    def clean_name(self, name: str) -> str:
        """清洗姓名"""
        name = unidecode(name)
        name = name.lower()
        new_name = ""
        for a in name:
            if a.isalpha():
                new_name += a
            else:
                new_name = new_name.strip()
                new_name += " "
        return new_name.strip()

    def get_name_uniq(self, name_c: str) -> float:
        """获取名字的稀有度得分"""
        name_rareness = 0.0
        if name_c:
            name_c = name_c.lower().split()
            for seg in name_c:
                s = self.name_uniq_dict.get(seg.strip(" "), 10)  # 未找到返回 10
                name_rareness += s
        return name_rareness

    def process_ranking_feature(self, ins: tuple) -> tuple[np.ndarray, float]:
        """计算 36 维特征

        Args:
            ins: tuple(paper_attr, author_paper_attr_list)
                paper_attr: (合作者集合, 机构, 期刊, 关键词, 标题)
                author_paper_attr_list: List[(合作者集合, 机构, 期刊, 关键词, 标题)]

        Returns:
            (特征向量, paper_coauthor_tfidf_ratio)
        """
        paper_attr, author_paper_attr_list = ins
        features = []
        name2clean = {}  # 姓名清洗缓存

        paper_names, paper_org, paper_venue, paper_keywords, paper_title = paper_attr
        paper_names = list(paper_names)[:50]  # 限制合作者数量

        # 处理姓名清洗
        for each in paper_names:
            if each not in name2clean:
                name2clean[each] = self.clean_name(each)

        # 收集候选作者信息
        candiauthor2int = defaultdict(int)  # 合作次数统计
        candiorgs = []
        candivenues = []
        candititles = []
        candikeywords = []

        filter_author_names = []
        for each in author_paper_attr_list:
            each_names, each_org, each_venue, each_keywords, each_title = each
            each_names = list(each_names)[:50]

            for name in each_names:
                tmp_clean = self.clean_name(name)
                candiauthor2int[tmp_clean] += 1
                if name not in name2clean:
                    name2clean[name] = tmp_clean

            filter_author_names.append(each_names)

            if each_org:
                candiorgs.append(each_org)
            if each_venue:
                candivenues.append(each_venue)
            if each_keywords:
                candikeywords.append(each_keywords)
            if each_title:
                candititles.append(each_title)

        # 计算合作者特征 (4维)
        paper_coauthor_tfidf_ratio = 0.0
        authorkeys = list(candiauthor2int.keys())

        if not paper_names or not authorkeys:
            features.extend([0.0] * 4)
        else:
            # 匹配合作者
            coauthors = set()
            for each_names in filter_author_names:
                each_coauthors = MatchName(paper_names, each_names, name2clean, True)
                coauthors = coauthors | each_coauthors

            # 计算 TF-IDF 加权得分
            coauthor_tfidf = 0.0
            counted_coauthor_tfidf = 0.0
            for each in coauthors:
                name_score = self.get_name_uniq(each)
                coauthor_tfidf += name_score
                counted_coauthor_tfidf += candiauthor2int.get(each, 1) * name_score

            paper_tfidf = 0.0
            for each in paper_names:
                name_score = self.get_name_uniq(name2clean[each])
                paper_tfidf += name_score

            author_tfidf = 0.0
            for each, count in candiauthor2int.items():
                name_score = self.get_name_uniq(each)
                author_tfidf += name_score * count

            paper_coauthor_tfidf_ratio = round(coauthor_tfidf / (paper_tfidf + 1e-8), 6)
            author_coauthor_tfidf_ratio = round(
                counted_coauthor_tfidf / (author_tfidf + 1e-8), 6
            )

            features.extend([
                coauthor_tfidf,
                paper_coauthor_tfidf_ratio,
                counted_coauthor_tfidf,
                author_coauthor_tfidf_ratio,
            ])

        # 计算其他特征 (机构、期刊、标题、关键词，各 8 维)
        features.extend(self._other_features(paper_org, candiorgs, DEFAULT_IDF_VALUES["org"]))
        features.extend(self._other_features(paper_venue, candivenues, DEFAULT_IDF_VALUES["venue"]))
        features.extend(self._other_features(
            normalize_title(paper_title),
            [normalize_title(t) for t in candititles],
            DEFAULT_IDF_VALUES["title"],
        ))
        features.extend(self._other_features(paper_keywords, candikeywords, DEFAULT_IDF_VALUES["keyword"]))

        return np.array(features), paper_coauthor_tfidf_ratio

    def _other_features(
        self,
        paper_attr: str,
        author_attr_list: list[str],
        default_value: float = 1.0,
    ) -> list[float]:
        """计算单组属性特征 (8维)

        Args:
            paper_attr: 待消歧论文的属性值
            author_attr_list: 候选作者历史论文的属性值列表
            default_value: 默认 IDF 值

        Returns:
            [max_jaro, mean_jaro, max_jaccard, mean_jaccard,
             score_paper, ratio_paper, score_author, ratio_author]
        """
        feature_list = []

        # 清洗文本
        paper_attr = " ".join(re.sub(r"[\W_]", " ", paper_attr).split())
        author_attr_list = [
            " ".join(re.sub(r"[\W_]", " ", item).split())
            for item in author_attr_list
        ]
        candi_string = " ".join(author_attr_list)

        if paper_attr.strip() and candi_string.strip():
            paper_attr_list = paper_attr.strip().lower().split()
            paper_attr_set = set(paper_attr_list)

            jaro_scores = []
            card_scores = []

            for item in author_attr_list:
                if item:
                    # Jaro-Winkler 相似度
                    jaros = distance.get_jaro_distance(paper_attr, item)
                    # Jaccard 相似度
                    item_set = set(item.split())
                    cards = len(item_set & paper_attr_set) / len(item_set | paper_attr_set)

                    jaro_scores.append(jaros)
                    card_scores.append(cards)

            # 统计值
            max_jaro_score = np.max(jaro_scores) if jaro_scores else 0.0
            mean_jaro_score = np.mean(jaro_scores) if jaro_scores else 0.0
            max_card_score = np.max(card_scores) if card_scores else 0.0
            mean_card_score = np.mean(card_scores) if card_scores else 0.0

            # TF-IDF 加权得分
            word_count_paper = defaultdict(int)
            for word in paper_attr_list:
                word_count_paper[word] += 1

            word_count_author = defaultdict(int)
            author_list = candi_string.strip().lower().split()
            for word in author_list:
                word_count_author[word] += 1

            inter_words = set(word_count_paper.keys()) & set(word_count_author.keys())

            score_paper = 0.0
            score_author = 0.0
            for inter in inter_words:
                # 使用对应的 IDF 字典
                if default_value == DEFAULT_IDF_VALUES["org"]:
                    score = self.org_tfidf.get(inter, default_value)
                elif default_value == DEFAULT_IDF_VALUES["venue"]:
                    score = self.ven_tfidf.get(inter, default_value)
                elif default_value == DEFAULT_IDF_VALUES["title"]:
                    score = self.title_tfidf.get(inter, default_value)
                else:
                    score = default_value

                score_paper += score * word_count_paper[inter]
                score_author += score * word_count_author[inter]

            total_score_paper = 0
            for word, count in word_count_paper.items():
                if default_value == DEFAULT_IDF_VALUES["org"]:
                    score = self.org_tfidf.get(word, default_value)
                elif default_value == DEFAULT_IDF_VALUES["venue"]:
                    score = self.ven_tfidf.get(word, default_value)
                elif default_value == DEFAULT_IDF_VALUES["title"]:
                    score = self.title_tfidf.get(word, default_value)
                else:
                    score = default_value
                total_score_paper += score * count

            total_score_author = 0
            for word, count in word_count_author.items():
                if default_value == DEFAULT_IDF_VALUES["org"]:
                    score = self.org_tfidf.get(word, default_value)
                elif default_value == DEFAULT_IDF_VALUES["venue"]:
                    score = self.ven_tfidf.get(word, default_value)
                elif default_value == DEFAULT_IDF_VALUES["title"]:
                    score = self.title_tfidf.get(word, default_value)
                else:
                    score = default_value
                total_score_author += score * count

            word_paper_ratio = round(score_paper / (total_score_paper + 1e-8), 6)
            # ratio_author 用归一化前的分子:交集占作者总词得分的比例,
            # 本身跨档案稳定(实测与 pub_count 相关仅 0.008),不应随归一化缩放
            word_author_ratio = round(score_author / (total_score_author + 1e-8), 6)

            # 归一化(7.2 根因①):score_author 随候选论文数近线性放大
            # (实测 ≈84/篇,pub_count 5→100 放大 10x),除以非空候选论文数
            n_candi_pubs = sum(1 for item in author_attr_list if item)
            if n_candi_pubs:
                score_author = round(score_author / n_candi_pubs, 6)

            feature_list.extend([
                max_jaro_score,
                mean_jaro_score,
                max_card_score,
                mean_card_score,
                score_paper,
                word_paper_ratio,
                score_author,
                word_author_ratio,
            ])
        else:
            feature_list.extend([0.0] * 8)

        return feature_list

    def extract(
        self,
        unassign_pub_info: dict,
        author_info: list[dict],
        name: str,
    ) -> tuple[np.ndarray, float]:
        """计算待消歧论文与候选作者之间的 36 维特征

        Args:
            unassign_pub_info: 待消歧论文信息
            author_info: 候选作者的论文信息列表
            name: 目标作者姓名

        Returns:
            (特征向量, paper_coauthor_tfidf_ratio)
        """
        from nallm.hand_feat.paper_attr import get_paper_attr

        author_paper_attrs = [
            get_paper_attr(pub_info, name)[0]
            for pub_info in author_info
        ]
        unassign_paper_attr = get_paper_attr(unassign_pub_info, name)[0]

        features, ratio = self.process_ranking_feature(
            (unassign_paper_attr, author_paper_attrs)
        )
        return features, ratio

    def feature(self, args: tuple) -> np.ndarray:
        """兼容旧接口"""
        return self.process_ranking_feature(args)[0]


__all__ = ["HandFeatureExtractor"]
