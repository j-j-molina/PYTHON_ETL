pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 90, unit: 'MINUTES')
    }

    triggers {
        pollSCM('H/5 * * * *')
    }

    parameters {
        string(name: 'USE_CASE', defaultValue: 'scoring_mora', description: 'Caso de uso configurado en src/config.json')
        string(name: 'EMAIL_TO', defaultValue: 'juanj.molina@upb.edu.co', description: 'Correo para notificación final')
    }

    environment {
        VENV_DIR = '.venv_jenkins'
        OUT_DIR  = 'jenkins_outputs'
    }

    stages {
        stage('Checkout SCM') {
            steps {
                checkout scm
                sh '''
                    set -e
                    mkdir -p "$WORKSPACE/${OUT_DIR}"
                    echo "[INFO] Repo descargado por Jenkins:"
                    pwd
                    ls -la
                '''
            }
        }

        stage('Verificar entorno Jenkins') {
            steps {
                sh '''
                    set -euo pipefail

                    command -v docker >/dev/null 2>&1 || { echo "[ERROR] docker no disponible"; exit 1; }
                    command -v git >/dev/null 2>&1 || { echo "[ERROR] git no disponible"; exit 1; }
                    command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 no disponible"; exit 1; }

                    [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || { echo "[ERROR] GOOGLE_APPLICATION_CREDENTIALS no definido"; exit 1; }
                    [ -f "${GOOGLE_APPLICATION_CREDENTIALS}" ] || { echo "[ERROR] No existe el archivo ADC en ${GOOGLE_APPLICATION_CREDENTIALS}"; exit 1; }

                    docker --version
                    git --version
                    python3 --version
                    docker ps >/dev/null
                '''
            }
        }

        stage('Validar estructura del proyecto') {
            steps {
                sh '''
                    set -eu

                    required_paths="
        src
        src/Cargar_datos.py
        src/comprension_eda.ipynb
        src/config.json
        src/ft_engineering.py
        src/heuristic_model.py
        src/model_training.py
        src/model_evaluation.py
        src/model_monitoring.py
        src/model_deploy.py
        tests
        tests/test_pipeline.py
        Dockerfile
        README.md
        requirements.txt
        requirements_deploy.txt
        run_pipeline.sh
        run_pipeline.bat
        set_up.sh
        set_up.bat
        sonar-project.properties
        "

                    missing=0

                    for path in $required_paths; do
                    if [ -e "$path" ]; then
                        echo "[OK] $path"
                    else
                        echo "[MISSING] $path"
                        missing=1
                    fi
                    done

                    if [ "$missing" -ne 0 ]; then
                    echo "[ERROR] La estructura del repositorio no cumple."
                    exit 1
                    fi

                    echo "[INFO] Estructura validada correctamente."
                '''
            }
        }

        stage('Instalar dependencias') {
            steps {
                sh '''
                    set -euo pipefail

                    rm -rf "${VENV_DIR}"
                    python3 -m venv "${VENV_DIR}"
                    . "${VENV_DIR}/bin/activate"

                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Ingesta desde GCP') {
            steps {
                sh '''
                    set -euo pipefail

                    . "${VENV_DIR}/bin/activate"
                    python src/Cargar_datos.py --use-case "${USE_CASE}"
                '''
            }
        }

        stage('Pruebas automatizadas') {
            steps {
                sh '''
                    set -euo pipefail

                    mkdir -p "$WORKSPACE/${OUT_DIR}/tests"
                    . "${VENV_DIR}/bin/activate"

                    pytest tests/ -v --tb=short \
                      --junitxml="$WORKSPACE/${OUT_DIR}/tests/test-results.xml" \
                      --cov=src \
                      --cov-report=xml:"$WORKSPACE/${OUT_DIR}/tests/coverage.xml"
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'jenkins_outputs/tests/test-results.xml'
                }
            }
        }

        stage('Ejecutar pipeline MLOps') {
            steps {
                sh '''
                    set -euo pipefail

                    mkdir -p "$WORKSPACE/${OUT_DIR}/pipeline"
                    . "${VENV_DIR}/bin/activate"

                    chmod +x run_pipeline.sh
                    bash run_pipeline.sh

                    cp -R artifacts "$WORKSPACE/${OUT_DIR}/pipeline/" || true
                    cp -R reports "$WORKSPACE/${OUT_DIR}/pipeline/" || true
                    cp -f Dockerfile "$WORKSPACE/${OUT_DIR}/pipeline/" 2>/dev/null || true
                    cp -f requirements_deploy.txt "$WORKSPACE/${OUT_DIR}/pipeline/" 2>/dev/null || true
                    cp -f .dockerignore "$WORKSPACE/${OUT_DIR}/pipeline/" 2>/dev/null || true
                '''
            }
        }
    }

    post {
        success {
            script {
                try {
                    mail(
                        to: params.EMAIL_TO,
                        subject: "✅ Jenkins SUCCESS | ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                        body: """El pipeline finalizó exitosamente.

Job: ${env.JOB_NAME}
Build: #${env.BUILD_NUMBER}
Use case: ${params.USE_CASE}
URL: ${env.BUILD_URL}
"""
                    )
                } catch (err) {
                    echo "[WARN] No se pudo enviar el correo de éxito: ${err}"
                }
            }
        }

        failure {
            script {
                try {
                    mail(
                        to: params.EMAIL_TO,
                        subject: "❌ Jenkins FAILURE | ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                        body: """El pipeline falló.

Job: ${env.JOB_NAME}
Build: #${env.BUILD_NUMBER}
Use case: ${params.USE_CASE}
URL: ${env.BUILD_URL}
"""
                    )
                } catch (err) {
                    echo "[WARN] No se pudo enviar el correo de fallo: ${err}"
                }
            }
        }

        always {
            archiveArtifacts allowEmptyArchive: true, artifacts: 'jenkins_outputs/**/*'
        }
    }
}