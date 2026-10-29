> Tarihli araştırma/audit kaydıdır. Güncel kod, bulgular ve çalışma durumu için [29 Ekim teslimini](FINAL_TR.md) ve [çalıştırma kılavuzunu](RUNBOOK_TR.md) okuyun. Bu kayıttaki eski hata ve süreç durumları bugünkü durumun beyanı değildir.

# HYDRA araştırma laboratuvarı

Bu klasör bağımsız araştırma ve salt okunur canlı gözlem içindir. HYDRA uygulaması ve Railway deploy'u değiştirilmedi. Emir gönderecek API, borsa hesabı veya işlem anahtarı kullanılmaz.

## Tamamlanan çalışma

- OKX BTC/USDT spot: 1 Ocak 2020–27 Ekim 2026, 59.802 kapanmış 1H mum. Veri kalite kontrolleri ve SHA-256: `data/quality.json`.
- Altı aile, 1H ve 4H, üç volatility exposure varyantı: **13.368 parametre seti**. Eğitimde farklılaşan **10.941 pozisyon yolu**. Bunlar bağımsız istatistiksel hipotezler değildir; çok sayıda benzer varyasyon içerir.
- 2020 Haziran–2023 geliştirme => 18 aday; 2024 doğrulama => 4 aday; seçilenler 2025 test ve 2026 tanısal dönemde değerlendirildi. Sonuçlara göre yeniden model seçilmedi.
- Temel maliyet her yönde %0,12; stres %0,22. İlave 1H işlem gecikmesi ayrıca denendi. Saatlik mark-to-market, başlangıç/final işlem maliyetleri, nakitte sıfır getiri, long/flat spot.
- **Canlı yön sinyaline terfi eden aday: 0.** Sonuç tablosu aşağıda. Mevcut örneklemin bir kısmı önceki audit'te görüldü; bu çalışma yeni ileriye dönük alfa kanıtı değildir.

| Aile / donmuş aday | 2024 doğrulama | 2025 net test | 2026 net tanısal | 2026 maliyet stresi |
|---|---:|---:|---:|---:|
| Kırılım `049b23e779f5ae78` | +%16,35 | +%5,92 | -%6,88 | -%8,89 |
| Çift EMA `806ef45dc42abb7a` | +%74,26 | +%6,05 | +%1,55 | -%2,24 |
| Momentum `3fce6fc229f42f60` | +%19,63 | -%26,58 | +%1,48 | -%1,12 |
| Trend `26be2fce8b7b8d3e` | +%69,85 | -%1,43 | +%0,23 | -%3,32 |

2025 al-tut yaklaşık -%6,58; 2026 tanısal dönem al-tut yaklaşık -%8,20. Bazı adaylar düşüşü azaltıyor; bu tek başına alfa kanıtı değildir. Pozitif excess'in belirsizlik aralıkları ve model aileleri için Holm düzeltmesi canlı terfi şartını sağlamadı. İşlem sayıları da çoğu adayda yetersiz. Her adayın reddedilme nedenleri `results/summary.json` içinde. Yaklaşık 2020–2023 getirileri güçlü görünse bile seçim yapılan dönemden alınan bu rakamlar beklenti olarak kullanılamaz.

## Deney sözleşmesi ve yeniden üretim

`results/manifest.json` tüm grid'i, veri hash'ini, dönemleri, sıralama ve terfi kurallarını kaydeder. `frozen-discovery-shortlist.json` ve `frozen-selected.json` son testten önce yazılır. 2025/2026 sonuçları model seçimine girmez. Aynı veri/spec ile tamamlanmış discovery tekrar kullanılabilir; farklı veri/spec için ayrı çıktı dizini oluşturulmalı. Başka aramalar yapıp yalnız en iyi sonucu raporlamak yasaktır.

```bash
cd /workspace/hydra-lab
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/hydra/bin/python -m unittest discover -v
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/hydra/bin/python check_results.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/hydra/bin/python run_search.py
```

17 test; zaman nedenselliği/prefix invariance, gelecekteki fiyatı değiştirme, sonraki açılış, işlem maliyeti, mark-to-market, günlük compounding, exposure sınırı, trade ID tekilliği, kontrat birimleri, gap backfill ve private endpoint reddi kapsanır. `tests.log` mevcut çalıştırma çıktısıdır.

Ana kanıtlar: `results/summary.json`, `leaderboard-discovery.csv`, `discovery.json`, `validation.json`, `evaluation-hourly.csv`, `search.log`. Veriler ve artefaktlar bu makinededir; Git'e push veya uzak ortama deploy yapılmadı.

## Gece gölge çalışması

Komut:

```bash
cd /workspace/hydra-lab
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/hydra/bin/python shadow.py --interval 30 --until 2026-10-28T05:00:00+00:00
```

Hedef duruş **28 Ekim 2026 08:00 Türkiye saati**. Çalışma yalnız mevcut cloud makinesinde başlatıldı; makine/oturum durursa kesintisiz çalışma garantisi yoktur. Durdurma saati geldiğinde süreç bir gözlem özeti yazar; kendi kendine canlı strateji açmaz. `morning_report.py` her zaman mevcut kayıtlarla tekrar özet üretebilir.

- OKX spot/perpetual işlemler, ticker, orderbook, OI, funding ve kapanmış saatlik mumlar kaydedilir.
- Trade ID primary key ile tekilleştirilir. Perpetual kontrat miktarı gerçek `ctVal` metadata ile BTC ve quote notional'a çevrilir.
- Poll pencereleri arasında bilinen trade ID bulunamazsa public history-trades ile en çok 20 sayfa geri doldurma denenir. Başarı teyit edilemezse gap açıkça raporlanır; eksik akış sağlıklı/tam sayılmaz.
- Kaynak payload ve event/fetched zamanı SQLite'ta saklanır; ham değer yaşlandıkça küçültülmez.
- Dört reddedilmiş aday yalnız **EXPERIMENTAL_RESEARCH_ONLY_NO_ORDERS** etiketiyle prospective forecast kaydı üretir. Başlangıç geçse `_LATE_STARTUP` olarak işaretlenir. Bunlar üretim sinyali değildir.
- Yeni 1H/4H/24H outcome yalnız gerçekten kaydedilmiş sonraki fiyat gözlemi varsa hesaplanır; henüz gelmemiş sonuçlar pending kalır. Bir gecelik örtüşen az sayıda gözlem alfa veya Sharpe kanıtı sayılmaz.

İzleme: `shadow-status.json`, `shadow.log`, `shadow.pid`, `data/shadow.sqlite3`. Sabah özeti: `MORNING_TR.md` ve `morning-summary.json`. İlk örnek ve kontrollü yeniden başlatma sırasında coverage gap/degraded kayıtları korunur; sonradan silinmez.

Manuel güncel özet:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/hydra/bin/python /workspace/hydra-lab/morning_report.py
```

## Bundan sonra

Milyarlarca benzer fiyat varyasyonuyla bu dört adayı tekrar tekrar seçmek yerine yeni bilgi taşıyan spot/perp akış ayrışması, OI ve funding hipotezi için point-in-time veri biriktirilmeli. Bu çalışma o veriyi toplamaya başladı. Sonraki deneylerde hipotez/parametre/dönem/terfi ölçütü önce kaydedilmeli; yeni sonuç görüldükten sonra eşik düşürülmemeli. Mevcut HYDRA'nın audit'te bulunan ham veri/freshness/ETF sorunları bu bağımsız laboratuvarla kendiliğinden düzelmiş değildir.
