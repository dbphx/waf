# WAF Model - HTTP Attack Detection (Go + ONNX)

This project implements a high-performance Web Application Firewall (WAF) detection model. It uses machine learning to identify HTTP attacks (SQLi, XSS, LFI, etc.) with native support for both Python and Golang runtimes.

## Core Achievements
- **100% Accuracy**: Passes the 210-category regression suite and 16 manual samples with zero false positives.
- **Native Go Support**: Native inference system implemented in Golang using ONNX Runtime for low-latency execution.
- **Bias Resilient**: Cleanly identifies root paths, short URIs, JWT tokens, and complex JSON as NORMAL traffic.

## Project Structure

```
.
├── application/go/          # REUSABLE WAF Libraries
│   └── logistic_regression/ # Go-native detector library
├── go/                      # CLI & Simulation Tool (standalone)
│   ├── internal/assets/     # Exported ONNX model & Metadata
│   ├── internal/features/   # Feature engineering logic
│   └── main.go              # CLI detector & simulation tool
├── src/                     # Training & Export (Python)
│   ├── standardize_data.py  # Advanced data pipeline
│   ├── train.py             # Logistic Regression training
│   ├── export_for_go.py     # ONNX export script
│   ├── test_categories.py   # 210-category regression suite
│   └── test_samples.py      # Manual validation suite
├── models/                  # Joblib originals (Scikit-Learn)
├── data/                    # Attack and Normal text datasets
└── README.md
```

## Golang Library Usage (Recommended)

The library at **`application/go/logistic_regression`** is the recommended way to integrate the WAF into your Go applications.

```go
import "logistic_regression"

// Initialize the detector
detector, err := logistic_regression.NewDetector(modelPath, metaPath, sharedLibPath)

// Predict from a map of request components
request := map[string]string{
    "path":  "/api/v1/user",
    "query": "id=1' OR '1'='1",
}
isAttack := detector.Predict(request)
```

## Golang CLI & Simulation

### 1. Build
Ensure you have the ONNX shared library on your system path.
```bash
cd go
go mod tidy
go build -o waf-detector main.go
```

### 2. Predict / Simulate
The Go binary supports raw HTTP simulation (Path, Query, Headers, Body parsing).
```bash
./waf-detector "POST /login?user=admin' OR '1'='1"
```

## Python Usage (Development)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain and Export
If you update `data/attack.txt`:
```bash
python3 src/standardize_data.py
python3 src/train.py
python3 src/export_for_go.py
```

### 3. Verify All Categories
```bash
python3 src/test_categories.py
```

## Performance Metrics
- **Regression Accuracy**: 100% (210/210 Categories)
- **Sample Accuracy**: 100% (16/16 Samples)
- **FPR**: 0% (Verified on current test suite)
- **Parity**: Python and Go results match bit-for-bit.

> [!TIP]
> Use the Go implementation (`go/main.go`) for production deployments as it provides lower latency and better memory efficiency via ONNX.
