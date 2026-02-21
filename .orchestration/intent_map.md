# Intent Map

This file maps high-level business intents to physical files and (optional) AST node identifiers.

- INT-001: Build Weather API
    - canonical_id: INT-001
    - description: "Primary intent to implement a weather data API and client adapters"
    - files:
        - path: src/weather/api.ts
          primary: true
          lines: 1-120
          note: "See trace: trace-20260221-0001"
        - path: src/weather/client.ts
          primary: false
          lines: 1-60
          note: "See trace: trace-20260221-0002"
        - path: tests/weather/
          primary: false
    - ast_nodes:
        - file: src/weather/api.ts
          node: FunctionDeclaration
          name: fetchWeather
          lines: 10-40
        - file: src/weather/api.ts
          node: ClassDeclaration
          name: WeatherService
          lines: 42-120
    - related_traces:
        - trace-20260221-0001
        - trace-20260221-0002

Update pattern: append new mappings when intent scope evolves. Keep entries minimal and human-readable.
