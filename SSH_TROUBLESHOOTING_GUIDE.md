# 🔧 SSH Troubleshooting Guide für Runpod TTS Training

Diese Anleitung behandelt **alle bekannten SSH-Probleme** und deren Lösungen beim Remote-Training auf Runpod.io.

## 🚨 Häufige SSH-Fehler und sofortige Lösungen

### 1. Permission denied (publickey)
**Fehler:** `git@github.com: Permission denied (publickey)`

**Sofortige Lösung:**
```bash
# 1. SSH Key erstellen und zur GitHub hinzufügen
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/github_key -N ""
cat ~/.ssh/github_key.pub | pbcopy  # Kopiere zur GitHub Settings > SSH Keys

# 2. SSH Config für GitHub
cat >> ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
```

### 2. Container SSH Key Probleme
**Fehler:** `Permission denied (publickey)` beim Container-Zugriff

**Sofortige Lösung:**
```bash
# 1. SSH Key für Runpod erstellen
./scripts/ssh_connect.sh setup-key

# 2. Public Key in Runpod einfügen
cat ~/.ssh/runpod-tts.pub
# Kopiere Output und füge in Runpod Settings > SSH Keys hinzu

# 3. Verbindung testen
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io
```

### 3. SSH Agent Probleme
**Fehler:** `Could not open a connection to your authentication agent`

**Sofortige Lösung:**
```bash
# SSH Agent starten
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_key
ssh-add ~/.ssh/runpod-tts

# Testen
ssh-add -l
```

## 🔍 Debug-Modus für SSH

### GitHub SSH Test
```bash
# GitHub Verbindung testen
ssh -T git@github.com

# Mit Debug-Output
ssh -vT git@github.com

# Mit sehr detailliertem Output
ssh -vvvT git@github.com
```

### Runpod SSH Test
```bash
# Runpod Verbindung testen
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io

# Mit Debug-Output
ssh -v -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io
```

## 🛠️ SSH Config Beispiele

### GitHub Config
```bash
# ~/.ssh/config für GitHub
cat > ~/.ssh/config << 'EOF'
# GitHub
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes

# Runpod
Host runpod-*
    HostName ssh.runpod.io
    IdentityFile ~/.ssh/runpod-tts
    IdentitiesOnly yes
EOF
```

### Multi-Key Setup
```bash
# Für mehrere Keys
cat > ~/.ssh/config << 'EOF'
# GitHub
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes

# Runpod Instance 1
Host runpod-tts1
    HostName ssh.runpod.io
    User POD_ID_1
    IdentityFile ~/.ssh/runpod-tts
    Port 22

# Runpod Instance 2
Host runpod-tts2
    HostName ssh.runpod.io
    User POD_ID_2
    IdentityFile ~/.ssh/runpod-tts
    Port 22
EOF
```

## 🔄 Git Clone mit SSH

### Repository klonen
```bash
# Mit HTTPS (falls SSH nicht funktioniert)
git clone https://github.com/AlJ95/finetuning.git

# Mit SSH (empfohlen)
git clone git@github.com:AlJ95/finetuning.git

# Falls SSH Key Passphrase
ssh-add ~/.ssh/github_key
git clone git@github.com:AlJ95/finetuning.git
```

## 📋 Schritt-für-Schritt SSH Setup Checkliste

### 1. Lokales Setup
```bash
# SSH Keys erstellen
ssh-keygen -t ed25519 -C "github" -f ~/.ssh/github_key -N ""
ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/runpod-tts -N ""

# Permissions setzen
chmod 700 ~/.ssh
chmod 600 ~/.ssh/*
chmod 644 ~/.ssh/*.pub

# SSH Config erstellen
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes

Host runpod-*
    HostName ssh.runpod.io
    IdentityFile ~/.ssh/runpod-tts
    IdentitiesOnly yes
EOF
```

### 2. GitHub Setup
```bash
# Public Key anzeigen
cat ~/.ssh/github_key.pub

# In GitHub einfügen:
# Settings → SSH and GPG keys → New SSH key
```

### 3. Runpod Setup
```bash
# Public Key anzeigen
cat ~/.ssh/runpod-tts.pub

# In Runpod einfügen:
# Runpod Settings → SSH Keys → Add SSH Key
```

## 🐛 Spezifische Fehlerbehebung

### Fehler: "Too many authentication failures"
```bash
# Ursache: SSH versucht zu viele Keys
# Lösung: IdentitiesOnly in SSH Config
cat >> ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes
EOF
```

### Fehler: "Host key verification failed"
```bash
# Ursache: Geänderter Host Key
# Lösung: Known hosts aktualisieren
ssh-keygen -R github.com
ssh-keygen -R ssh.runpod.io
```

### Fehler: "Connection timed out"
```bash
# Ursache: Netzwerk/Firewall
# Lösung: Port prüfen
telnet ssh.runpod.io 22
# Oder alternative Ports testen
```

## 🚀 Automatisierte SSH Setup Scripts

### Setup-Script für neue Systeme
```bash
cat > setup_ssh_complete.sh << 'EOF'
#!/bin/bash
echo "🔧 SSH Setup für Runpod TTS Training..."

# SSH Keys erstellen
if [ ! -f ~/.ssh/github_key ]; then
    ssh-keygen -t ed25519 -C "github" -f ~/.ssh/github_key -N ""
    echo "✅ GitHub SSH Key erstellt"
fi

if [ ! -f ~/.ssh/runpod-tts ]; then
    ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/runpod-tts -N ""
    echo "✅ Runpod SSH Key erstellt"
fi

# SSH Config erstellen
cat > ~/.ssh/config << 'CONFIG'
# GitHub
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes

# Runpod
Host runpod-*
    HostName ssh.runpod.io
    IdentityFile ~/.ssh/runpod-tts
    IdentitiesOnly yes
CONFIG

# Permissions setzen
chmod 700 ~/.ssh
chmod 600 ~/.ssh/*
chmod 644 ~/.ssh/*.pub
chmod 600 ~/.ssh/config

echo "🎯 SSH Setup abgeschlossen!"
echo "📋 GitHub Key: $(cat ~/.ssh/github_key.pub)"
echo "📋 Runpod Key: $(cat ~/.ssh/runpod-tts.pub)"
EOF

chmod +x setup_ssh_complete.sh
```

## 📊 SSH Verbindung testen

### Kompletter Test-Workflow
```bash
# 1. SSH Agent starten
eval "$(ssh-agent -s)"

# 2. Keys hinzufügen
ssh-add ~/.ssh/github_key
ssh-add ~/.ssh/runpod-tts

# 3. GitHub testen
ssh -T git@github.com

# 4. Runpod testen
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io

# 5. Git Clone testen
git clone git@github.com:AlJ95/finetuning.git test_clone
```

## 🆘 Notfall-Lösungen

### Fallback zu HTTPS
```bash
# Falls SSH komplett nicht funktioniert
git clone https://github.com/AlJ95/finetuning.git
cd finetuning
# Dann manuelles Setup
```

### Manuelle Datei-Übertragung
```bash
# Falls SSH nicht funktioniert
# 1. Files hochladen via Runpod Web Interface
# 2. Oder SCP mit Passwort (falls aktiviert)
scp -r ./finetuning root@YOUR_IP:/workspace/
```

## 📝 Checkliste für SSH Probleme

- [ ] SSH Keys erstellt
- [ ] Public Keys zu GitHub/Runpod hinzugefügt
- [ ] SSH Config korrekt
- [ ] Permissions korrekt (600 für private keys)
- [ ] SSH Agent läuft
- [ ] Keys zum Agent hinzugefügt
- [ ] Verbindung getestet

## 🔗 Weitere Ressourcen

- [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [Runpod SSH Documentation](https://docs.runpod.io/ssh/)
- [SSH Config Examples](https://www.ssh.com/academy/ssh/config)
