#!/bin/bash
# SSH Connection Script für Runpod.io - Finale Version
# Behebt alle SSH/Terminal Probleme

# Konfiguration
RUNPOD_SSH_KEY="${HOME}/.ssh/runpod-tts"

print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# SSH Key Setup
setup_ssh_key() {
    if [[ ! -f "$RUNPOD_SSH_KEY" ]]; then
        print_info "Erstelle SSH Key für Runpod..."
        ssh-keygen -t ed25519 -C "runpod-tts" -f "$RUNPOD_SSH_KEY" -N ""
        print_info "SSH Key erstellt: $RUNPOD_SSH_KEY"
        print_info "Füge folgenden Public Key zu Runpod hinzu:"
        cat "${RUNPOD_SSH_KEY}.pub"
    fi
}

# Einfache Verbindung
connect_ssh() {
    local pod_id=$1
    [[ -z "$pod_id" ]] && { print_error "Pod ID erforderlich"; return 1; }
    ssh -t -i "$RUNPOD_SSH_KEY" "$pod_id@ssh.runpod.io"
}

# Setup mit einzelnen Schritten
setup_remote() {
    local pod_id=$1
    [[ -z "$pod_id" ]] && { print_error "Pod ID erforderlich"; return 1; }
    
    print_info "=== Runpod TTS Setup Anleitung ==="
    echo ""
    echo "1. Verbindung herstellen:"
    echo "   ssh $pod_id@ssh.runpod.io -i ~/.ssh/runpod-tts"
    echo ""
    echo "2. Auf der Remote Instance ausführen:"
    echo "   cd /workspace"
    echo "   git clone https://github.com/AlJ95/finetuning.git"
    echo "   cd finetuning"
    echo "   chmod +x scripts/*.sh"
    echo "   ./scripts/setup_runpod.sh"
    echo ""
    echo "3. Training starten:"
    echo "   python scripts/download_datasets.py --datasets thorsten --thorsten-size 1000"
    echo "   python scripts/train_remote.py --dataset thorsten --epochs 3"
}

# Training starten (interaktiv)
start_training() {
    local pod_id=$1
    local dataset=${2:-thorsten}
    local epochs=${3:-3}
    
    [[ -z "$pod_id" ]] && { print_error "Pod ID erforderlich"; return 1; }
    
    print_info "Starte interaktives Training..."
    print_info "Verbinde zu $pod_id..."
    print_info "Führe folgende Befehle aus:"
    echo ""
    echo "cd /workspace"
    echo "git clone https://github.com/AlJ95/finetuning.git"
    echo "cd finetuning"
    echo "./scripts/setup_runpod.sh"
    echo "python scripts/download_datasets.py --datasets $dataset --${dataset}-size 1000"
    echo "python scripts/train_remote.py --dataset $dataset --epochs $epochs"
    echo ""
    echo "Verbindung wird hergestellt..."
    
    ssh -t -i "$RUNPOD_SSH_KEY" "$pod_id@ssh.runpod.io"
}

# Hauptfunktion
main() {
    case "$1" in
        setup-key)
            setup_ssh_key
            ;;
        connect)
            connect_ssh "$2"
            ;;
        setup)
            setup_remote "$2"
            ;;
        train)
            start_training "$2" "$3" "$4"
            ;;
        *)
            echo "Usage: $0 {setup-key|connect|setup|train}"
            echo ""
            echo "Commands:"
            echo "  setup-key              - SSH Key erstellen"
            echo "  connect <pod-id>       - SSH Verbindung"
            echo "  setup <pod-id>         - Setup Anleitung anzeigen"
            echo "  train <pod-id> [dataset] [epochs] - Interaktives Training"
            echo ""
            echo "Beispiel:"
            echo "  $0 connect jacxa38ckzy1ye-64411204"
            echo "  $0 setup jacxa38ckzy1ye-64411204"
            echo "  $0 train jacxa38ckzy1ye-64411204 thorsten 3"
            ;;
    esac
}

# Skript ausführen
main "$@"
