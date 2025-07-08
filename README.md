# 🇹🇷 Türkçe Metin Özetleyici (mT5 Tabanlı)

Bu proje, HuggingFace Transformers kütüphanesi ve `mT5` (Multilingual T5) modeli kullanılarak, Türkçe metinlerin otomatik olarak özetlenmesini sağlar. Kullanıcıdan alınan serbest bir metin, model tarafından analiz edilir ve içerdiği bilginin kısa bir özeti üretilir.

---


### 1. Introduction (Giriş)

Doğal Dil İşleme (NLP) alanında metin özetleme, uzun metinleri daha kısa, anlamlı ve bilgi açısından yoğun hale getirmek amacıyla sıkça kullanılan bir tekniktir. Özellikle haber, hukuk, sağlık ve akademik metinlerde içerik yoğunluğunu azaltarak kullanıcıya hızlı bilgi edinimi sağlar. Bu projede, çok dilli destek sunan `mT5` (Multilingual T5) modeliyle Türkçe metinlerin özetlenmesi amaçlanmıştır.

---

### ⚙️ 2. Methods (Yöntemler)

Proje Python dilinde geliştirilmiş olup, HuggingFace'in `transformers` kütüphanesi kullanılmıştır. Kullanılan model:  
`csebuetnlp/mT5_multilingual_XLSum`

#### Kullanılan teknolojiler:
- Python 3.10+
- HuggingFace Transformers
- PyTorch
- SentencePiece

#### Uygulama Adımları:
1. Kullanıcıdan Türkçe serbest metin alınır.
2. Metin, tokenizer aracılığıyla `input_ids` formatına çevrilir.
3. `mT5` modeli bu token'ları işler ve özet çıkarır.
4. Token çıktısı tekrar anlamlı Türkçe cümleye dönüştürülür.

#### Kurulum:
```bash
pip install transformers torch sentencepiece

---
### Results (Sonuçlar)


