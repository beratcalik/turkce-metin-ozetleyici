# 🇹🇷 Türkçe Metin Özetleyici (mT5 Tabanlı)

Bu proje, HuggingFace Transformers kütüphanesi ve `mT5` (Multilingual T5) modeli kullanılarak, Türkçe metinlerin otomatik olarak özetlenmesini sağlar. Kullanıcıdan alınan serbest bir metin, model tarafından analiz edilir ve içerdiği bilginin kısa bir özeti üretilir.

---


### 1. Introduction (Giriş)

Doğal Dil İşleme (NLP) alanında metin özetleme, uzun metinleri daha kısa, anlamlı ve bilgi açısından yoğun hale getirmek amacıyla sıkça kullanılan bir tekniktir. Özellikle haber, hukuk, sağlık ve akademik metinlerde içerik yoğunluğunu azaltarak kullanıcıya hızlı bilgi edinimi sağlar. Bu projede, çok dilli destek sunan `mT5` (Multilingual T5) modeliyle Türkçe metinlerin özetlenmesi amaçlanmıştır.

---

### 2. Methods (Yöntemler)

Proje Python dilinde geliştirilmiş olup, HuggingFace'in `transformers` kütüphanesi kullanılmıştır. Kullanılan model:  
`csebuetnlp/mT5_multilingual_XLSum`

#### 1. Gerekli Kütüphanelerin Kurulumu:
```bash
pip install transformers torch sentencepiece
```
#### 2. Model ve Tokenizer'ın Yüklenmesi

```python
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
```
-MT5Tokenizer: Kullanıcının girdiği metni modele uygun sayısal formata (token'lara) çevirir.

-MT5ForConditionalGeneration: Verilen input'a karşılık bir output (özet) üretir.

#### 3. Kullanıcıdan Metin Alma

```python
metin = input("📥 Özetlenecek metni giriniz: ")
```
-Kullanıcıdan serbest biçimli bir Türkçe metin alınıyor.

#### 4. Tokenize Etme (Sayılara Çevirme)

```python
inputs = tokenizer(metin, return_tensors="pt", max_length=512, truncation=True)
```
-return_tensors="pt": Çıktının PyTorch tensörü olmasını sağlar.

-max_length=512: Maksimum token uzunluğu, model sınırıdır.

-truncation=True: Uzun metinleri otomatik olarak keser.

#### 5.Model ile Özet Üretme

```python
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=60,
    min_length=20,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
```
-max_length / min_length: Üretilecek özetin uzunluk sınırları.

-length_penalty=2.0: Daha kısa ve öz cümleler üretmesini sağlar.

-num_beams=4: Beam Search algoritması ile daha iyi sonuçlar üretir.

-early_stopping: Uygun uzunlukta durmasını sağlar.

#### 6. Model Çıktısını Metne Dönüştürme
```python
ozet = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

```

-Token ID'leri tekrar anlaşılır Türkçe cümleye çevrilir.

-skip_special_tokens=True özel sembolleri (`<pad>`, `<s>`) filtreler.

#### 7. Özetin Ekrana Yazdırılması

```python
print("\n📌 Özet:\n", ozet)
```

-Bu satır, kullanıcının verdiği metne karşılık üretilen özeti terminalde gösterir.


### 3. Results (Sonuçlar)

Model, girilen metinlerdeki temel fikirleri başarılı bir şekilde yakalayarak 1-2 cümlelik anlamlı özetler üretmektedir. Aşağıda örnek bir sonuç verilmiştir:

#### Girdi(Metin)

Yapay zeka, günümüzde birçok sektörde devrim yaratmaktadır. Sağlık alanında teşhis süreçlerini hızlandırmakta, finans sektöründe ise dolandırıcılıkları önlemeye yardımcı olmaktadır. Eğitimde kişiselleştirilmiş öğrenme deneyimleri sunarken, üretim sektöründe robotlar sayesinde verimliliği artırmaktadır.

#### Model Çıktısı (Özet):

Yapay zeka, sağlık, finans ve eğitim gibi alanlarda verimliliği artırmakta ve önemli değişimler sağlamaktadır.

### 4.Discussion (Tartışma):

mT5 modeli, çok dilli eğitim verisi sayesinde Türkçe dilinde de oldukça başarılı özetleme sonuçları vermektedir. Ancak bazı sınırlamalar da mevcuttur:

-Çok uzun metinlerde model giriş sınırını (512 token) aşarsa bazı bilgiler kırpılabilir.

-Noktalama işaretleri eksik olan metinlerde anlam kaymaları olabilir.

-Modelin özet uzunluğu sabittir, dinamik kontrol zordur.


Gelecekte bu projeye şu özellikler eklenebilir:

-Web arayüzü (Streamlit veya Flask)

-Özetin bilgi yoğunluğu skoruyla değerlendirilmesi

-Çoklu metin yükleme ve toplu özetleme

-Farklı model karşılaştırmaları (T5, BART, Pegasus, vb.)

### Hazırlayan
Berat Çalık
Ankara Üniversitesi - Yapay Zeka ve Veri Mühendisliği
