# Cüzdan yönü: tek hesap etkisi kontrolü

2026-10-29T10:16:45.280630+03:00

Kullanılabilir çeyrek saat: 81; bir hesabı çıkarınca aggregate yönünü koruyamayan: 40.

| Tanım | Yön gözlemi | 15m ortalama brüt | 60m ortalama brüt | 4h ortalama brüt |
|---|---:|---:|---:|---:|
| aggregate | 81 | -0.003% | +0.029% | +0.187% |
| equal_account_breadth | 63 | -0.034% | -0.107% | -0.207% |
| robust_aggregate | 41 | +0.005% | +0.015% | -0.016% |

Bu yeni kontrol geçmişte toplanmış veriye sonradan uygulanmıştır. Aynı sonuçları ileriye dönük başarı diye sunmaz. Yönü sıfır olan hesap dengesi abstain sayılır. Fiyat hareketleri fill, funding/slippage sonrası getiri veya bağımsız olay kanıtı değildir. Cüzdan/parametre seçimi PnL ile yapılmaz; gerçek emir yoktur.
