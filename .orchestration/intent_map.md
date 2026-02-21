# Intent Map

This file maps high-level business intents to physical files and (optional) AST node identifiers.

- INT-001: Build Weather API
    - files:
        - src/weather/api.ts # primary implementation
        - src/weather/client.ts # HTTP client and adapters
        - tests/weather/\*
    - ast_nodes:
        - src/weather/api.ts:FunctionDeclaration:fetchWeather
        - src/weather/api.ts:ClassDeclaration:WeatherService

Update pattern: append new mappings when intent scope evolves. Keep entries minimal and human-readable.
