# 29 Ekim — onay sonrası ek araştırma

Bu aşama ilk teslimi genişletir. Gerçek emir, ücretli API, bildirim, Git push veya Railway yayını yapılmadı.

- Public recentTrades katılımcılarından 95 ek adresin mevcut hesabı incelendi.
- PnL veya gelecekteki performans kullanılmadan, önceden açıklanmış üç BTC büyüklük grubundan **16 yeni hesap** seçilip donduruldu: 6 küçük, 6 orta, 4 büyük.
- Önceki 6 hesap aynen korunur; iki örneklemde toplam **22 farklı public hesap** var. Bağımsız sahip sayısı bilinmiyor.
- İlk gerçek API kontrolünde **16/16 hesap** ve 838 pozisyon satırı okundu. Kaynak hatası ve fill sayfa kesilmesi yoktu. Bu satır sayısı bağımsız işlem veya piyasa kapsamı sayılmaz.
- Yeni örneklem yaklaşık $217,2M BTC gross pozisyon gösteriyor; **%96,7 tek hesaba ait**. Cüzdan sayısının artması, dolar ağırlıklı görüşün çeşitlendiğini göstermedi.

İlk sözleşme ve yeni adresler [kanıt dizininde](evidence/continued-20261029/additional-frozen-cohort.json).
Yeni hesaplar eski deneyin geçmişine eklenmez; yeni örneklemin gelecekteki gözlemleri ayrı SQLite journal'a yazılır.

## Tek hesap etkisi kontrolü

[Kontrol raporu](BREADTH_TR.md), önceki kayıtların 81 kullanılabilir çeyrek saatini inceler.
**40 gözlemde**, hesaplardan biri çıkarılınca toplam envanter değişiminin yönü korunmuyor.
Eşit hesap yönü ile notional/toplam baz yönü aynı değil. Tek hesabı çıkarmaya dayanıklı
41 gözlemde ortalama yönlü brüt hareket 15m +%0,005, 60m +%0,015, 4h -%0,016;
24 bp gidiş-geliş maliyet eşiğine dayanıklı avantaj gösterilmedi.

Bu karşılaştırmalar geçmişte toplanmış veriye sonradan uygulandı. Yeni bir başarılı
OOS veya kilitlenmiş ileriye dönük strateji testi diye sunulmuyor. Ortak sahiplik,
market-maker envanteri ve dış hedge'ler bilinmiyor; hesap sayısı istatistiksel
bağımsızlık sayılmaz. Geçmişte kötü olan yönü ters çevirerek yeni kazanan seçilmedi.

## Çalışan düzen

Üç bounded salt okunur süreç: OKX/EMA, önceki 6 hesap ve ayrı 16 hesap örneklemi.
Hedef duruş **30 Ekim 2026 08:00 Türkiye saati**; ortam açık kalmalıdır.
Yeni worker 240 saniye poll aralığıyla API bütçesini sınırlar. İlk yaklaşık
15 dakikadaki envanter farkı bilinmiyor olarak kalır; sıfırla doldurulmaz.

`GET /api/research` ve Research Evidence ek örneklemi ayrı gösterir; örneklemler
arasında hesapları tekilleştirir. Liquidation fill hacimleri farklı örneklemlerden
toplanıp bütün piyasa hacmi gibi gösterilmez. Mevcut 6 hesap ve eski donmuş arşiv
değiştirilmedi. Hesap yönü veya tahmini likidasyon seviyesi üretim sinyaline bağlanmadı.

## Erişim sınırı

Kullanıcı onayından sonra Pain, HypurrScan, Nansen web ve Hyperliquid GitBook erişimi
yeniden denendi. Proxy tüneli hâlâ **403 Forbidden** döndürüyor. Onay kaydedilmiş
taslağı çalışan ağ politikasına otomatik uygulamadı. Kullanılabilir ortam araçları
taslak okuma/kaydetme sağlıyor; yayımlama işlemi sunmuyor. Ağ değişikliği Codex ortam
arayüzünde uygulandıktan sonra doğrudan ürün incelemesi tamamlanabilir. Engellenen
arayüzler incelenmiş veya fiyatları doğrulanmış gibi sunulmuyor.

Tekrar üretim ve worker komutları [çalıştırma kılavuzundadır](RUNBOOK_TR.md).
Bu aşama için 148 test geçti; iki mevcut deprecation uyarısı var. Araştırma API'si
22 hesabı doğru ayırarak yanıt verdi; frontend JavaScript syntax kontrolü geçti.
