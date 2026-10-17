# Procesador de imágenes RGB · Ecualización interactiva

Aplicación web para **ecualizar progresivamente los canales R, G y B de una imagen de forma
independiente**, con **cuatro sliders (0–100 %)** y **ocho histogramas** (original y actualizado
por cada vista).

- **Frontend:** Astro (JavaScript en el navegador: aplica las tablas de transformación y dibuja los histogramas).
- **Backend:** FastAPI (Python) con NumPy y Pillow: calcula histogramas, tablas de transformación y el procesamiento de referencia.

---

## Requisitos

| Herramienta | Versión recomendada |
|---|---|
| Node.js | 22 o superior |
| Python | 3.10 o superior |

## Cómo levantarlo

Necesitas **dos terminales**: una para el backend y otra para el frontend.

### 1. Backend (puerto 8000)

```powershell
cd backend

# Instalar dependencias (solo la primera vez)
python -m pip install -r requirements.txt

# Levantar el servidor
python -m uvicorn main:app --port 8000
```

Debe quedar escuchando en `http://127.0.0.1:8000`.

### 2. Frontend (puerto 4321)

```powershell
cd frontend

# Instalar dependencias (solo la primera vez)
npm install

# Levantar el servidor de desarrollo
npm run dev
```

Abre el navegador en `http://localhost:4321`.

> La URL del backend es configurable. Por defecto apunta a `http://127.0.0.1:8000`.
> Para cambiarla, crea un archivo `frontend/.env` con:
> `PUBLIC_API_URL=http://mi-servidor:8000`

---

## Qué hace el sistema

1. **Carga una imagen**: al abrir la página se carga automáticamente una *imagen de ejemplo* de bajo
   contraste (generada por el backend), o puedes **subir tu propia imagen** (JPG, PNG, BMP, etc.).

2. **Ecualización interactiva**: cuatro deslizadores de **0 % a 100 %**:

   | Slider | Efecto |
   |---|---|
   | **General (RGB)** | Sincroniza los tres canales con el mismo porcentaje |
   | **Rojo (R)** | Ecualiza solo el canal R |
   | **Verde (G)** | Ecualiza solo el canal G |
   | **Azul (B)** | Ecualiza solo el canal B |

   Los sliders **no suman brillo**: controlan el progreso de una transformación del histograma.
   Cuando los canales tienen porcentajes distintos, el slider general muestra **«Mixto»** y la línea
   inferior indica `R 60% · G 20% · B 100%`. El botón **Reiniciar** devuelve los cuatro controles a 0 %.

3. **Algoritmo (por canal, de forma independiente)**: para cada canal se calcula su histograma de
   256 niveles, su CDF y la tabla de ecualización completa `T_eq`. Después se interpola con el
   porcentaje `p` del slider (normalizado 0–1):

   ```
   T_p(k) = (1 - p) · k + p · T_eq(k)
   ```

   con redondeo al entero más cercano y recorte a `[0, 255]`. `T_eq` se deriva siempre del
   **histograma original** (nunca de una imagen ya modificada), por lo que el procesamiento **no
   acumula** transformaciones: el 0 % devuelve la imagen exacta original.

   ```python
   cdf     = histograma.acumulado()
   cdf_min = primer acumulado no nulo
   T_eq(k) = floor((cdf[k] - cdf_min) / (N - cdf_min) * 255)
   T_p(k)  = round((1 - p) * k + p * T_eq(k))
   ```

   Si la imagen es constante (no hay variación suficiente) se usa la transformación identidad.

4. **Ocho histogramas** con datos reales de los píxeles (256 niveles, misma escala en cada par
   original/actualizado):

   | # | Panel | | # | Panel |
   |---|---|---|---|---|
   | 1 | RGB original (R+G+B superpuestos) | | 5 | G original |
   | 2 | RGB actualizado | | 6 | G actualizado |
   | 3 | R original | | 7 | B original |
   | 4 | R actualizado | | 8 | B actualizado |

5. **Vista previa y descarga**: se ven las dos imágenes lado a lado (original/procesada) y el
   botón **Descargar** guarda el resultado en PNG.

### Estrategia de actualización

La imagen se envía al backend **una sola vez** (`POST /api/prepare`), que devuelve la imagen
original, sus histogramas y las tres tablas `T_eq` de 256 entradas. A partir de ahí cada
movimiento de slider se procesa en el navegador (milisegundos, sin viajes a la red), y los ocho
histogramas se recalculan sobre los píxeles reales de la imagen de salida.

---

## Estructura del proyecto

```
Deber imagen/
├── backend/
│   ├── main.py                        # API FastAPI: /api/sample, /api/prepare, /api/process
│   ├── image_utils.py                 # Histogramas, LUTs T_eq, interpolación y procesamiento RGB
│   ├── requirements.txt
│   └── tests/
│       ├── conftest.py
│       └── test_equalization.py       # 24 pruebas (pytest)
├── frontend/
│   ├── src/
│   │   ├── lib/equalize.js            # Lógica pura: LUTs, píxeles, histogramas, estado de sliders
│   │   └── pages/index.astro          # 4 sliders, 8 histogramas, vista previa
│   ├── tests/
│   │   ├── equalize.test.mjs          # 11 pruebas (node --test)
│   │   └── integration.test.mjs       # integración frontend <-> backend (si el backend está activo)
│   ├── public/
│   ├── astro.config.mjs
│   └── package.json
└── README.md
```

## API del backend

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/sample` | Devuelve la imagen de ejemplo (PNG de bajo contraste). |
| `POST` | `/api/prepare` | Recibe la imagen y devuelve la imagen original en base64, los histogramas originales de R/G/B y las tablas `T_eq` de cada canal (se calculan una sola vez). |
| `POST` | `/api/process` | Recibe la imagen + `eq_r`, `eq_g`, `eq_b` (0–100) y devuelve antes/después en base64 con los histogramas original y actualizado. |
| `GET` | `/docs` | Documentación interactiva automática de FastAPI. |

### Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:8000/api/process \
  -F "file=@mi_imagen.png" \
  -F "eq_r=60" \
  -F "eq_g=20" \
  -F "eq_b=100"
```

---

## Pruebas

### Backend (24 pruebas)

```powershell
cd backend
python -m pip install pytest httpx     # solo la primera vez
python -m pytest tests -q
```

Cubren los 15 puntos del enunciado: 0 % idéntico a la original, interpolación al 50 %,
ecualización completa al 100 %, independencia de cada canal, sincronización del slider general,
estados mixtos, actualización de los ocho histogramas, histogramas originales inmutables,
coincidencia de los histogramas actualizados con los píxeles de salida, independencia del orden,
imágenes constantes, formatos de entrada (PNG/JPEG/BMP, RGBA, escala de grises) y validaciones.

### Frontend (12 pruebas)

```powershell
cd frontend
npm test
```

Verifican la lógica pura de `src/lib/equalize.js` (interpolación de LUTs, aplicar 0 %, ejemplos
por canal, orden de movimientos, sincronización y estado «Mixto», histogramas reales) y la
estructura de la página (exactamente 4 sliders 0–100 y 8 histogramas, sin controles de brillo).
La prueba de integración se omite automáticamente si el backend no está levantado.

---

## Problemas comunes

- **La página muestra un error tipo “¿Está corriendo el backend…?”**
  El backend no está levantado o la URL no coincide. Verifica que `uvicorn` esté corriendo en el
  puerto `8000`.
- **El puerto ya está en uso**
  Cambia el puerto (`uvicorn main:app --port 8001`) y ajusta `PUBLIC_API_URL` en el frontend.
- **La imagen no se procesa**
  Verifica que sea un formato soportado (JPG, PNG, BMP, etc.) y que no supere los 20 MB.
- **Las pruebas de integración del frontend aparecen como omitidas**
  Levanta el backend antes de ejecutar `npm test`.
