# HYDRA — tamamlanan çalışma ve güncel kanıt

Güncel ek çalışma: [29 Ekim onay sonrası ilerleme](CONTINUATION_20261029_TR.md).

Yeni ekosistem taraması: [16 ürün ve 3 açık kaynak araç](TOOLS_CATALOG_TR.md).

Rapor: 2026-10-29T09:19:07.224764+03:00

**Karar: doğrulanmış, maliyetlere dayanıklı alfa henüz bulunmadı.** Yeni araştırma modülleri üretim sinyallerini değiştirmiyor. Emir gönderimi kapalı.

Tarihsel tarama: 13,368 varyasyon; 10,941 farklı eğitim pozisyon yolu; test öncesi dondurulan 4 aday; kabul ölçütlerini geçen 0.

OKX: 2,694,734 tekil işlem; 3,467 döngünün 210 tanesinde kalite/coverage sorunu.
Gözlem aralığı: 2026-10-27T19:57:31.637000+00:00 — 2026-10-29T04:59:50.662000+00:00.
Aynı donmuş EMA/spot sözleşmesiyle 22 karar; spot doğrulaması 19 kararda kullanılabilir. Baseline giriş sayısı 0. İşlemsizlik getirisi alfa sayılmaz.

Hyperliquid: 6 donmuş hesap, 525 snapshot; 237,018 tekil fill; 144 açık liquidation fill'i (60 BTC).
Gözlem aralığı: 2026-10-28T09:34:08.334000+00:00 — 2026-10-29T04:59:57.202000+00:00.
Fill API sayfa sınırına ulaşılan 7 döngü var; bu pencerelerin bütün fill'leri içerdiği varsayılmaz.
Son BTC örnekleminde tek hesabın payı %78.8; gross ağırlıklarına göre etkin hesap sayısı 1.57. Bu, bütün piyasanın veya bağımsız akıllı paranın görüşü değildir.

BTC fill'leri erişim dakikası/yönüne göre 12 gruba indi. İki spot penceresinde ters yön desteği 0 grupta var; coverage bilinmeyen 5 grup ayrı tutuldu.

| Ufuk | Gözlenen grup | Örtüşmeyen aralık | Tüm ters yönler: ortalama brüt hareket | Spot desteği olanlar: ortalama brüt hareket |
|---|---:|---:|---:|---:|
| 15m | 12 | 5 | +0.149% | ölçülmedi |
| 60m | 12 | 3 | +0.014% | ölçülmedi |
| 240m | 4 | 2 | +0.088% | ölçülmedi |

Bunlar erişim zamanı korunarak yapılmış geriye dönük açıklayıcı ölçümlerdir; kilitlenmiş ileriye dönük strateji testi değildir. Örtüşmeyen aralıklar da istatistiksel bağımsızlık garantilemez. Brüt fiyat hareketi; gerçekleşmiş kâr, short/spot portföy getirisi veya funding/slippage içeren backtest değildir. Örneklem bütün likidasyonları kapsamaz. Gidiş-geliş 24 bp maliyet eşiği JSON'da ayrıca gösterilir; Sharpe/p-değeri üretilmez.

Cüzdan baz envanter yönü: 77 kullanılabilir çeyrek saat gözlemi; en erken erişilebilir snapshot seçildi, fiyat etkisi baz miktardan ayrıldı.

| Ufuk | Gözlenen yön | Örtüşmeyen aralık | Ortalama yönlü brüt fiyat hareketi |
|---|---:|---:|---:|
| 15m | 76 | 56 | -0.001% |
| 60m | 73 | 16 | +0.029% |
| 240m | 61 | 4 | +0.192% |

Bu örneklemde envanter yönünün ortalama hareketi de 24 bp maliyet eşiğini geçmedi.

Kısa gözlem süresi ve yoğunlaşmış cohort ile performans genellenemez.

Araştırma önceliği: gerçekleşen likidasyon + spot absorpsiyonu; ardından aynı donmuş hesapların baz envanter değişimi. Sonuçlara bakarak eşik veya cüzdan seçimi değiştirilmedi.

Rakip ve kaynak analizi: `docs/research/COMPETITORS_TR.md`. Pain ve bazı ticari arayüzler ortamın ağ kısıtı nedeniyle incelenemedi; fiyat/özellik doğrulaması tamamlanmış sayılmaz.

Yerel kod ve raporlar hazırdır. Railway yayını ve sürüm eşleşmesi doğrulanmadı. Eski ML simülatörü doğrulanmış değil; ML kapalı kalır.

Uygulama teslimi: backtest motoru, seçim doğrulayıcısı, iki public collector ve erişim-zamanlı replay repo içindedir. Donmuş 13 dosyalık paket hash kontrolüyle çevrimdışı geri yüklenip bağımsız dizinde doğrulandı. API ve Research Evidence alanı bulguları tarihli gösterir. Veri/cache/ETF/OI/funding düzeltmelerine ek olarak yanıltıcı CVD ve günlük faiz ölçüleri, kaynak tarihi ve doğrulanmamış takvim etiketleri düzeltildi.

Doğrulama: **131 test geçti**, iki dependency/test deprecation uyarısı var; `pip check` başarılı. Gerçek OKX ile bir uygulama döngüsü ve araştırma API'si dahil yedi HTTP kontrolü geçti. Frontend JavaScript syntax kontrolü geçti. GitHub Actions test workflow'u eklendi; uzaktaki CI çalışması yapılmış gibi raporlanmıyor.

Tekrar çalıştırma: [operasyon kılavuzu](RUNBOOK_TR.md). Ayrıntılı JSON: [kanıt snapshot'ı](evidence/latest-summary.json). Yerel değişiklikler incelemeye hazırdır; Git push, merge ve Railway deploy yapılmadı.

Yeni salt okunur gözlem turu, eski journal dosyalarının ayrı kopyaları üzerinde başlatıldı; ilk yeniden başlama boşluğu silinmedi ve spot doğrulaması unknown kaldı. Ardından iki OKX trade akışında ID örtüşmesi görüldü; Hyperliquid ilk çevrimde 6/6 hesap okudu. Hedef duruş **30 Ekim 2026 08:00 Türkiye saati**; ortam açık kalmalıdır. [Doğrulama kaydı](evidence/validation.json).
