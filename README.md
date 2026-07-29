# GigaAM v3 — CoreML encoder for Apple Neural Engine

## ⬇️ Скачать / Download

**[gigaam-v3-encoder-ane.mlpackage.zip · 408 МБ](https://github.com/IsaacClarke2/gigaam-v3-coreml/releases/download/v3.0/gigaam-v3-encoder-ane.mlpackage.zip)**
— модель лежит в [Releases](https://github.com/IsaacClarke2/gigaam-v3-coreml/releases), в дереве репозитория только скрипт конвертации.
The model lives in Releases (git trees don't like 400 MB binaries); the repo itself holds the conversion script.

---

Энкодер [GigaAM v3](https://github.com/salute-developers/GigaAM) (SberDevices, MIT) в формате
CoreML MLProgram fp16 — работает на Neural Engine маков Apple Silicon.
Конвертирован из [ONNX-версии istupakov](https://huggingface.co/istupakov/gigaam-v3-onnx)
скриптом `convert-gigaam-ane.py` (onnx2torch → coremltools), веса не менялись.

The [GigaAM v3](https://github.com/salute-developers/GigaAM) (SberDevices, MIT) speech encoder
as CoreML MLProgram fp16 for the Apple Neural Engine. Converted 1:1 from
[istupakov's ONNX export](https://huggingface.co/istupakov/gigaam-v3-onnx) with
`convert-gigaam-ane.py` — weights untouched.

## Бенчмарки: ONNX против CoreML / Benchmarks

**Железо:** MacBook Pro 14″, M1 Pro (8P+2E), 16 ГБ, macOS 26.1.
**Материал:** 23-минутная запись живой разговорной русской речи (16 кГц моно) и 30-секундный кусок из неё.
**Метод:** `/usr/bin/time` — стена и процессорное время; CoreML — после разовой
ANE-специализации (~40 c при первой загрузке, дальше системный кэш).
Hardware: M1 Pro, 16 GB. Material: a 23-minute recording of real conversational Russian speech. Wall / CPU time
via `/usr/bin/time`; CoreML measured after the one-time ANE specialization.

### Только энкодер / Encoder only (окно 33.6 c, паддинг нулями)

| Вариант | 30 c звука | Скорость |
|---|---|---|
| ONNX Runtime, CPU fp32, 6 потоков | 1.31 c | 1× |
| CoreML fp16, `CPU_ONLY` | 0.29 c | 4.5× |
| **CoreML fp16, `CPU_AND_NE`** | **0.13 c** | **10×** |

### Полный конвейер / Full RNNT pipeline (запись 23:45: лог-мел → энкодер → жадный RNNT)

| Конфигурация | Стена / Wall | CPU-время / CPU time |
|---|---|---|
| Всё на ONNX Runtime (CPU) | 69.2 c | 325.5 c |
| CoreML-энкодер (ANE) + ONNX-декодер | 33.6 c | 52.2 c |
| **CoreML-энкодер (ANE) + декодер на Accelerate** | **24.9 c** | **17.7 c** |

Итого к исходному ONNX-конвейеру: **стена ×2.8, процессорная работа ×18**.
CPU-версия занимала все performance-ядра на минуты (вентиляторы);
ANE-версия — ~1 занятое ядро на полминуты, в мониторинге почти не видна.
Net vs the all-ONNX baseline: **2.8× wall, 18× less CPU work** — the CPU
path pegged every P-core and spun the fans, the ANE path is near-silent.

### Диктовка: короткие фразы / Dictation-style latency

Полный путь «звук → текст» (лог-мел → энкодер → жадный RNNT), модель
РЕЗИДЕНТНА, медиана из 5 тёплых прогонов; фразы вырезаны из той же живой
записи. Full audio-to-text latency, resident model, median of 5 warm runs.

| Фраза | ONNX CPU fp32 (6 потоков) | CoreML ANE fp16 | ANE выигрыш |
|---|---|---|---|
| 2 c | **92 мс** | 148 мс | 0.6× |
| 5 c | 188 мс | **161 мс** | 1.2× |
| 10 c | 354 мс | **195 мс** | 1.8× |
| 30 c | 1141 мс | **294 мс** | 3.9× |

Честный нюанс: до ~4 секунд ONNX быстрее — у него динамическая длина
входа, а ANE всегда платит за полное окно 33.6 c. Дальше ANE уходит в
отрыв, и его задержка почти не растёт (150–300 мс на любую фразу —
предсказуемость, которую любит диктовка). И второе измерение, которого
нет в таблице: ONNX на каждый вызов занимает шесть потоков
performance-ядер, ANE почти не трогает CPU — для диктовки на батарее
это важнее миллисекунд.

The honest trade-off: below ~4 s ONNX wins (dynamic input length vs the
fixed 33.6 s ANE window); past that ANE pulls ahead and stays nearly
flat (150–300 ms for any phrase). And the axis the table doesn't show:
every ONNX call burns six P-core threads, the ANE call barely touches
the CPU — on battery that matters more than milliseconds.

### Качество / Accuracy (fp16 vs fp32)

| Метрика | Значение |
|---|---|
| Расхождение слов на выборке 339 слов живой речи | 8 слов (2.4%), замены равноценные: «прям»↔«прямо», пунктуация |
| Вклад одного паддинга до окна (fp32↔fp32) | 0.6% |
| Косинус выходов энкодера (валидная зона) | 0.989 |
| Токены на записи 23:45 | 5941 → 5888 (−0.9%) |

⚠️ CoreML Execution Provider внутри самого ONNX Runtime на этом графе
НЕ работает (компилируется, падает на выполнении, «error −1», проверено
с MLProgram) — потому и понадобилась прямая конвертация.
Note: ONNX Runtime's own CoreML EP fails at execution on this graph —
hence the direct conversion.

## Формат / Format

- Вход / inputs: `audio_signal` `[1, 64, 3360]` float32 — лог-мел
  (окно Ханна 320, хоп 160, 64 мела; 3360 кадров = 33.6 c, короткие куски
  паддятся нулями), `length` `[1]` int32 — честная длина в кадрах.
- Выход / outputs: `encoded` `[1, 768, 840]` float32, `encoded_len` `[1]` int32
  (⚠️ на паддинге завышен на 1 — считайте `ceil(length / 4)` сами).
- Окно ФИКСИРОВАННОЕ: ANE не поддерживает динамические формы.
- Декодер и joint RNNT в комплект не входят — они крошечные (4.6 + 2.7 МБ),
  берите ONNX у istupakov, CPU хватает с запасом.
- Первая загрузка на машине ~40 c (ANE-специализация), дальше — системный кэш.

## Использование / Usage (Swift)

```swift
import CoreML

// .mlpackage компилируется один раз; .mlmodelc можно закэшировать рядом.
let compiled = try await MLModel.compileModel(at: packageURL)
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: compiled, configuration: config)

let audio = try MLMultiArray(shape: [1, 64, 3360], dataType: .float32)
// … заполните лог-мел фичами (паддинг нулями до 3360 кадров),
// пишите через .strides — CoreML не гарантирует плотную укладку!
let length = try MLMultiArray(shape: [1], dataType: .int32)
length[0] = NSNumber(value: Int32(validFrames))

let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
    "audio_signal": MLFeatureValue(multiArray: audio),
    "length": MLFeatureValue(multiArray: length),
]))
// out: "encoded" [1, 768, 840] — читать тоже через .strides.
```

## Квантизация / Quantization

Меряли data-free сжатие весов той же методикой (расхождение слов против
fp32-эталона на 339-словной выборке живой речи). We measured data-free
weight compression with the same word-diff methodology vs the fp32 reference.

| Вариант | Размер | Расхождение слов | Задержка (5 c / 30 c) | Вердикт |
|---|---|---|---|---|
| fp16 (базовый) | 423 МБ | 2.4% | 161 / 294 мс | ✅ рекомендуем |
| **int8 linear per-channel** | **212 МБ** | **2.4%** | 155 / 293 мс | ✅ **бесплатно по качеству** |
| 6-bit palettization (grouped/16) | 162 МБ | 4.1% | — | ⚠️ на грани |
| 4-bit palettization (grouped/16) | 108 МБ | 14.5% | — | ❌ разваливает текст |
| 4-bit palettization (per-tensor) | 107 МБ | 17.7% | — | ❌ |

**Выводы:** int8 — половина размера бесплатно
([`gigaam-v3-encoder-ane-int8.mlpackage.zip`](https://github.com/IsaacClarke2/gigaam-v3-coreml/releases/download/v3.0/gigaam-v3-encoder-ane-int8.mlpackage.zip),
⚠️ требует macOS 15+). Честный q4 без потерь этой модели data-free
не даётся — нужна калибровка/QAT. TL;DR: int8 halves the size for free
(macOS 15+); honest q4 needs calibration or QAT — data-free it wrecks
the transcript.

## Лицензия / License

MIT — наследована от GigaAM (© SberDevices) и ONNX-экспорта istupakov.
