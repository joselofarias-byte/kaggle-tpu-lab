# Integración Android / eventos en español

Esta rama define un contrato simple para que una app Android pueda mostrar el estado real de una sesión TPU sin tener que interpretar texto libre.

## Objetivos

- Distinguir claramente **en cola**, **iniciando**, **lista**, **error** y **detenida**.
- Mostrar el diagnóstico real cuando algo falla.
- Mantener compatibilidad con los consumidores existentes: `phase` y los campos históricos siguen presentes.
- Añadir mensajes en español listos para UI.

## Evento ntfy

Los kernels Qwen y GLM publican JSON con este formato base:

```json
{
  "event_version": 1,
  "phase": "compiling",
  "state": "starting",
  "message_es": "Compilando gráficos XLA (4 min transcurridos)."
}
```

Campos:

- `event_version`: versión del contrato.
- `phase`: fase técnica existente.
- `state`: estado normalizado para interfaz.
- `message_es`: texto corto para mostrar al usuario.
- El resto de los campos dependen de la fase y se conservan por compatibilidad.

## Estados normalizados

| state | Uso recomendado en Android |
| --- | --- |
| `starting` | Iniciando / preparando / compilando |
| `ready` | TPU lista y endpoint disponible |
| `error` | Falló el arranque o el servicio |
| `stopped` | Fin normal por tiempo máximo |

La cola de Kaggle ocurre antes de que el kernel pueda publicar eventos; la app debe conservar un estado adicional **queued** obtenido desde la API de Kaggle.

La cabecera global de la app no debe mostrar `Running` mientras la cuenta siga en `QUEUED`.

## Errores

Los eventos `failed` y `stopped` pueden incluir:

```json
{
  "phase": "failed",
  "state": "error",
  "step": "no-tpu",
  "error_code": "no-tpu",
  "recoverable": true,
  "cause": "...",
  "hint_es": "...",
  "tail": "...",
  "message_es": "Kaggle inició la sesión sin una TPU utilizable. Verificá la cuenta/acelerador y volvé a lanzar."
}
```

Orden sugerido para la UI:

1. `message_es`
2. `hint_es`
3. `cause`
4. `tail` dentro de un panel expandible **Detalles**

Nunca reemplazar estos campos por un genérico "Unknown cause" si existe cualquiera de ellos.

## Caso no-tpu

Este caso apareció durante una prueba real: Kaggle creó la ejecución, pero el contenedor veía solamente CPU.

La app debe mostrarlo como un error recuperable y no como un fallo de Qwen/vLLM.

Si la cuenta todavía no está habilitada para TPU, el usuario debe completar la verificación exigida por Kaggle. Si ya está habilitada, conviene detener y volver a lanzar.

## Sonda de salud

Un evento `ready` no demuestra que el túnel siga respondiendo. La app y `launch.py status` consultan `GET {endpoint}/v1/models` con el bearer de la sesión. La clave no se escribe en el error.

- Mientras la sonda responde 200, el estado global sigue en **TPU lista**.
- Tras montar el endpoint se esperan 45 s antes de contar fallos, para no marcar **Sin conexión** durante el arranque del túnel.
- Tres fallos seguidos pasan ese endpoint a `OFFLINE` y la cabecera muestra **Sin conexión** aunque el kernel de Kaggle siga en `RUNNING`.
- Una sonda exitosa posterior vuelve a **TPU lista** sin relanzar el kernel.

## API key en ntfy

`api_key` viaja solo en el evento `ready`, porque la adopción de una sesión en cola que la app no creó lo necesita. El resto de las fases lo omiten. Los logs de pip y de vLLM lo redactan. El banner `READY` de la celda del notebook sigue mostrándolo para quien está mirando esa salida.

## Secretos

- No mostrar el token personal de Kaggle en logs.
- La API key del endpoint puede copiarse desde la UI, pero no debe registrarse completa en diagnósticos.
- En Android, almacenar el token con Android Keystore o almacenamiento cifrado. Si se implementa fallback, debe ser explícito y nunca borrar la única copia válida hasta verificar persistencia después de reiniciar el proceso.

## Compatibilidad

Los nuevos campos son aditivos. Clientes anteriores pueden seguir consumiendo únicamente `phase`, `endpoint`, `api_key`, `cause`, `hint`, `tail`, etc.
