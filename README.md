# Algoritmos Metaheuristicos

Este repositorio alberga los trabajos y desarrollos realizados en la materia de Metaheurísticas. Aquí encontrarás implementaciones de diversos algoritmos metaheurísticos, prácticas y proyectos relacionados con la optimización y resolución de problemas diversos.

El objetivo principal es explorar, implementar y analizar diferentes técnicas metaheurísticas como algoritmos genéticos, optimización por enjambre de partículas, y otros métodos inspirados en la naturaleza, aplicándolos a problemas de optimización en distintas áreas.

## 📁 Estructura del Repositorio

* 📂  **files** : Contiene archivos de datos y otros recursos utilizados en los proyectos.
* 📂  **mylib** : Incluye módulos y librerías desarrollados para la reutilización de funciones y clases a lo largo de los proyectos.
* 📂  **notebooks** : Carpeta con notebooks de Jupyter donde se desarrollan y prueban los algoritmos metaheurísticos.
* 📄  **requirements.txt** : Lista de dependencias necesarias para ejecutar los notebooks y scripts del repositorio.
* ⚙️  **setup.py** : Archivo de configuración para la instalación de las librerías personalizadas en `mylib`.

## 🔥 Tutorial de Configuración del Proyecto Metaheurísticas

### 1. 🚀 Clonar el Repositorio

Primero, necesitas clonar el repositorio de GitHub en tu máquina local. Abre la terminal o línea de comandos y ejecuta el siguiente comando:

```bash
git clone https://github.com/Leodna/metahueristicos.git
```

También puedes descargar el proyecto en un archivo comprimido .zip

![Descargar zip del proyecto](files\readme\descarga_proy.png)

### 2. 📂 Navegar al Directorio del Proyecto

Una vez clonado el repositorio, navega a la carpeta del proyecto.

### 3. 🐍 Crear un Entorno Virtual con Python 3.8 usando Anaconda

Para gestionar las dependencias de manera aislada, se recomienda crear un entorno virtual con Anaconda específico para Python 3.8.
1. Crear el entorno virtual con Python 3.8:

```bash
conda create --name metah python=3.8
```
2. Activar el entorno virtual:
```bash
conda activate metaheuristicas
```

### 4. 📦 Instalar las Dependencias

Con el entorno virtual activado, instala las dependencias necesarias que están listadas en el archivo requirements.txt:

Con el entorno virtual activado, instala todas las dependencias listadas en requirements.txt:
```bash
pip install -r requirements.txt
```

### 5. ⚙️ Configurar el Entorno de Trabajo

Si es necesario, puedes instalar la librería personalizada del proyecto usando el archivo setup.py:

```bash
python setup.py install
```

O también puedes ejecutar, esto si te encuentras en la carpeta donde descargaste el repositorio:
```bash
pip install -e .
```

### 6. 🧪 Ejecutar el Proyecto

Ahora estás listo para ejecutar los notebooks o scripts del proyecto. Abre Jupyter Notebook o cualquier otra herramienta de desarrollo para empezar.

## 🤝 Colaboración

Este proyecto es gestionado de manera colaborativa para fomentar el aprendizaje y la mejora continua. ¡Explora, contribuye y aprende con nosotras mientras aplicamos técnicas avanzadas de optimización!
