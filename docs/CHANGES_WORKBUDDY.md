# 本机改动记录（WorkBuddy）

> 上游：[maverickxone/notion2api](https://github.com/maverickxone/notion2api)（MIT）
> 本分支在上游基础上做了「模型同步」和「本机易用性」两件事，没有改动反代核心协议。

## 1. 从线上流量里挖出模型代号表

上游注册表停在 `claude-opus5 = agave-flan`。实际代号表由 Notion 服务端下发：

```
POST https://app.notion.com/api/v3/getAppConfig
→ initialValues.dynamic_configs[*].value.modelCardInfo   # 48 个模型：代号 → 显示名
   （另有 zeroRetentionModels 给出同样的映射）
```

抓法：起一个带 `--remote-debugging-port` 的 Chrome，注入已登录的 `token_v2`
（`Network.setCookie`），打开 `/ai`，用 `Network.getResponseBody` 抓 `getAppConfig`。

## 2. 新增 16 个模型（后端共 39 个）

| 外部名 | Notion 代号 | 显示名 |
|---|---|---|
| **claude-opus5.5** | **albuquerque-quinn** | Opus 5.5 |
| claude-sonnet5.5 | achira-donut | Sonnet 5.5 |
| claude-haiku5.5 | amla-jam | Haiku 5.5 |
| gpt-6-sol | onico | GPT-6 Sol |
| gpt-6.1-sol | omniberry-sundae | GPT-6.1 Sol |
| gpt-6-luna | omodi | GPT-6 Luna |
| gpt-5.2 | oatmeal-cookie | GPT-5.2 |
| gemini-3.7flash | grapefruit-zeppole | Gemini 3.7 Flash |
| kimi-k3 | fireworks-kimi-k3 | Kimi K3 |
| deepseek-v4flash | baseten-deepseek-v4-flash | DeepSeek V4 Flash |
| glm-5.3 | baseten-glm-5.3 | GLM 5.3 |
| glm-5.3flash | baseten-glm-5.3-flash | GLM 5.3 Flash |
| grok-4.6 | soursop-shortcake | Grok 4.6 |
| grok-4.7 | grok-4.7 | Grok 4.7 |
| fable-5.1 | assam-chai | Fable 5.1 |
| muse-spark-1.2 | muse-spark-1.2 | Muse Spark 1.2 |

改动文件（按上游 `docs/ADD_MODEL.md` 的清单）：
`app/model_registry.py`、`frontend/index.html`、`frontend/js/core/constants.js`。

## 3. 本机易用性脚本

- `refresh_token.py` — 直接解密本机 Chrome Cookie 库拿新的 `token_v2`，回写 `accounts.json`。
  token 过期后只需在 Chrome 重新登录 Notion，再跑一次即可，不必重填 space 信息。
  解密链路：keychain `Chrome Safe Storage` → PBKDF2(saltysalt, 1003, 16)
  → AES-CBC(v10 前缀，明文从偏移 16 起，尾部按 PKCS7 去填充)。
- `start.sh` / `stop.sh` — 起停 uvicorn（127.0.0.1:8000）。
