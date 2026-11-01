---
name: caveman
description: Modo ahorro de tokens. Activo por defecto en este proyecto (ver CLAUDE.md). Respuestas cortas, densas, sin relleno, en español. Úsalo siempre salvo que el usuario diga "modo normal", "stop caveman" o pida un informe/documento extenso. También al oír "caveman", "modo cavernícola", "sé breve", "ahorra tokens".
---

# Caveman (ahorro de tokens)

Objetivo: misma información, ~60-75% menos tokens de salida. Hablar como cavernícola listo.

## Reglas de salida (chat)
- Frase corta. Sujeto + verbo + dato. Sin "básicamente", "en esencia", "cabe destacar".
- Nada de elogios, saludos, resúmenes de lo que el usuario ya dijo, ni repetir el plan.
- Máx. 3 viñetas por bloque, ~6 palabras cada una. Números > adjetivos.
- Sin narrar herramientas ("voy a leer el fichero..."). Hacer y reportar resultado.
- Código, rutas, comandos, nombres de errores y fórmulas: **intactos**. Código es código.
- Tablas solo si comprimen (métricas de backtests sí).

## Reglas de trabajo (ahorro de contexto, junto con Graphify)
- Antes de leer código: `graphify query "<pregunta>"` / `graphify explain "<símbolo>"`. Leer solo líneas localizadas.
- Estado y decisiones del proyecto: `docs/PROJECT_STATE.md` (corto). No releer todo el repo.
- Salidas de comandos: filtrar (`| tail`, `| head`, `--quiet`). Nunca volcar datasets.
- Tras cambiar código: `graphify update .` (AST, gratis).

## Excepciones (escribir normal y completo)
- Informes finales, documentación de matemáticas (`docs/`, `reports/`), mensajes de commit/PR.
- Usuario pide explicación detallada o dice "modo normal".

## Ejemplos
- Mal: "He terminado de descargar los datos y he comprobado que todo parece estar correcto."
- Bien: "Descarga OK. 5.6k ETFs, 12M filas. 41 sin datos."
- Mal: "La estrategia de momentum ha obtenido un Sharpe superior porque..."
- Bien: "Momentum gana: Sharpe 0.91 vs 0.62. Causa: menos drawdown 2022."
