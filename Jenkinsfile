pipeline {
    agent any

    // ──────────────────────────────────────────────
    // Trigger: se ejecuta automáticamente en cada
    // push / merge a la rama Master
    // ──────────────────────────────────────────────
    triggers {
        pollSCM('H/5 * * * *')  // Revisa cambios en GitHub cada 5 minutos
        // Si tienes webhook configurado en GitHub, agrega también:
        // githubPush()
    }

    environment {
        PYTHON    = 'python3'
        VENV_DIR  = 'venv'
        EMAIL_TO  = 'tu-correo@ejemplo.com'   // ← cambia por tu correo
    }

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {

        // ── 1. CLONAR REPOSITORIO ─────────────────
        stage('Checkout') {
            steps {
                echo '📥 Clonando repositorio desde GitHub...'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/Master']],
                    userRemoteConfigs: [[
                        url: 'https://github.com/j-j-molina/mlops_pipeline.git'
                        // Si el repo es privado agrega:
                        // credentialsId: 'github-credentials'
                    ]]
                ])
            }
        }

        // ── 2. VALIDAR ESTRUCTURA DE CARPETAS ─────
        // PRUEBA REQUERIDA: verifica que el repo tenga
        // todos los archivos y carpetas necesarios
        stage('Validar estructura de carpetas') {
            steps {
                dir('..') {   // ← sube un nivel como indica la guía del curso
                    echo '🗂️  Verificando estructura del proyecto...'
                    script {
                        def requiredPaths = [
                            'src',
                            'src/Cargar_datos.py',
                            'src/ft_engineering.py',
                            'src/heuristic_model.py',
                            'src/model_training.py',
                            'src/model_evaluation.py',
                            'src/model_deploy.py',
                            'src/model_monitoring.py',
                            'src/config.json',
                            'data/raw',
                            'data/state',
                            'reports',
                            'tests',
                            'requirements.txt',
                            'Dockerfile'
                        ]

                        def missing = []
                        requiredPaths.each { path ->
                            if (!fileExists(path)) {
                                missing << path
                            }
                        }

                        if (missing) {
                            error("❌ Faltan archivos/carpetas:\n  - ${missing.join('\n  - ')}")
                        } else {
                            echo '✅ Estructura de carpetas correcta.'
                        }
                    }
                }
            }
        }

        // ── 3. INSTALAR DEPENDENCIAS ──────────────
        stage('Instalar dependencias') {
            steps {
                dir('..') {
                    echo '🐍 Instalando dependencias...'
                    sh """
                        ${PYTHON} -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    """
                }
            }
        }

        // ── 4. TESTS UNITARIOS ────────────────────
        stage('Tests') {
            steps {
                dir('..') {
                    echo '🧪 Ejecutando pruebas unitarias...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        pytest tests/ --tb=short --junitxml=reports/test-results.xml -v
                    """
                }
            }
            post {
                always {
                    junit 'reports/test-results.xml'
                }
            }
        }

        // ── 5. CARGAR DATOS ───────────────────────
        stage('Cargar datos') {
            steps {
                dir('..') {
                    echo '📊 Cargando datos...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        ${PYTHON} src/Cargar_datos.py
                    """
                }
            }
        }

        // ── 6. FEATURE ENGINEERING ────────────────
        stage('Feature Engineering') {
            steps {
                dir('..') {
                    echo '⚙️  Ejecutando feature engineering...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        ${PYTHON} src/ft_engineering.py
                    """
                }
            }
        }

        // ── 7. ENTRENAMIENTO ──────────────────────
        stage('Entrenamiento del modelo') {
            steps {
                dir('..') {
                    echo '🤖 Entrenando modelo...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        ${PYTHON} src/model_training.py
                    """
                }
            }
        }

        // ── 8. EVALUACIÓN ─────────────────────────
        stage('Evaluación del modelo') {
            steps {
                dir('..') {
                    echo '📈 Evaluando modelo...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        ${PYTHON} src/model_evaluation.py
                    """
                }
            }
        }

        // ── 9. DEPLOY ─────────────────────────────
        stage('Deploy') {
            steps {
                dir('..') {
                    echo '🚀 Desplegando modelo...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        ${PYTHON} src/model_deploy.py
                    """
                }
            }
        }

        // ── 10. MONITOREO ─────────────────────────
        stage('Monitoreo') {
            steps {
                dir('..') {
                    echo '📡 Ejecutando monitoreo...'
                    sh """
                        . ${VENV_DIR}/bin/activate
                        ${PYTHON} src/model_monitoring.py
                    """
                }
            }
        }
    }

    // ──────────────────────────────────────────────
    // NOTIFICACIÓN AL TERMINAR (requerimiento fase 2)
    // ──────────────────────────────────────────────
    post {
        success {
            echo '✅ Pipeline finalizado exitosamente.'
            mail(
                to: "${EMAIL_TO}",
                subject: "✅ [Jenkins] mlops_pipeline #${BUILD_NUMBER} — ÉXITO",
                body: """
Build exitoso.
Proyecto  : ${JOB_NAME}
Build #   : ${BUILD_NUMBER}
Duración  : ${currentBuild.durationString}
Ver resultado: ${BUILD_URL}
                """
            )
        }

        failure {
            echo '❌ Pipeline fallido.'
            mail(
                to: "${EMAIL_TO}",
                subject: "❌ [Jenkins] mlops_pipeline #${BUILD_NUMBER} — FALLO",
                body: """
Build fallido.
Proyecto  : ${JOB_NAME}
Build #   : ${BUILD_NUMBER}
Ver logs  : ${BUILD_URL}console
                """
            )
        }

        unstable {
            echo '⚠️ Pipeline inestable — revisa los tests.'
            mail(
                to: "${EMAIL_TO}",
                subject: "⚠️ [Jenkins] mlops_pipeline #${BUILD_NUMBER} — INESTABLE",
                body: "Revisa los resultados: ${BUILD_URL}testReport"
            )
        }

        always {
            echo '🧹 Limpiando workspace...'
            cleanWs()
        }
    }
}
