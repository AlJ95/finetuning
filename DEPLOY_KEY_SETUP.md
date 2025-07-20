# 🔑 Deploy Key Setup für GitHub Repository

Diese Anleitung zeigt die **korrekte Einrichtung des Deploy Keys** für GitHub Zugriff auf Runpod.

## 🎯 Warum Deploy Key?

- **Sicherer** als persönliche SSH Keys
- **Repository-spezifisch** - nur Zugriff auf dieses Repo
- **Automatisiert** - kein Passwort erforderlich
- **Best Practice** für Cloud-Instances

## 📋 Schritt-für-Schritt Deploy Key Setup

### 1. Deploy Key auf Remote Instance erstellen

**Nach SSH Verbindung zur Runpod Instance:**
```bash
# SSH zur Runpod Instance verbinden
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io

# Deploy Key erstellen
ssh-keygen -t ed25519 -C "deploy-key-runpod" -f ~/.ssh/github_key -N ""

# Public Key anzeigen und kopieren
cat ~/.ssh/github_key.pub
```

### 2. Deploy Key in GitHub einrichten

**Auf GitHub:**
1. Gehe zu: https://github.com/AlJ95/finetuning/settings/keys
2. Klicke: **"Add deploy key"**
3. **Titel:** `Runpod Deploy Key`
4. **Key:** Füge den Inhalt von `~/.ssh/github_key.pub` ein
5. **✅ Allow write access** aktivieren
6. Klicke: **"Add key"**

### 3. SSH Config auf Remote Instance

```bash
# SSH Config für GitHub erstellen
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes
EOF

# Permissions setzen
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/github_key
chmod 644 ~/.ssh/github_key.pub
```

### 4. Repository mit Deploy Key klonen

```bash
# Repository klonen (mit Deploy Key)
git clone git@github.com:AlJ95/finetuning.git
cd finetuning

# Setup ausführen
chmod +x scripts/*.sh
./scripts/setup_runpod.sh
```

## ✅ Verifizierung

```bash
# Teste GitHub Verbindung
ssh -T git@github.com
# Erwartete Ausgabe: "Hi AlJ95/finetuning! You've successfully authenticated..."

# Teste Clone
git clone git@github.com:AlJ95/finetuning.git test_clone
cd test_clone && ls -la
```

## 🚨 Fehlerbehebung

### Fehler: "Permission denied (publickey)"
```bash
# SSH Agent starten
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_key

# Teste Verbindung
ssh -vT git@github.com
```

### Fehler: "Host key verification failed"
```bash
# Known hosts aktualisieren
ssh-keygen -R github.com
ssh -T git@github.com
```

## 🔄 Alternative: HTTPS mit Token

**Falls Deploy Key nicht funktioniert:**
```bash
# HTTPS mit Personal Access Token
git clone https://github.com/AlJ95/finetuning.git
cd finetuning
./scripts/setup_runpod.sh
```

## 📋 Komplette Befehlsübersicht

```bash
# Gesamter Workflow nach SSH Verbindung
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io

# Auf Remote Instance:
ssh-keygen -t ed25519 -C "deploy-key-runpod" -f ~/.ssh/github_key -N ""
cat ~/.ssh/github_key.pub  # Kopiere diesen Key
# Füge Key in GitHub ein (siehe oben)

cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
git clone git@github.com:AlJ95/finetuning.git
cd finetuning
chmod +x scripts/*.sh
./scripts/setup_runpod.sh
```

## 🎯 Erfolgs-Checkliste

- [ ] Deploy Key erstellt auf Remote Instance
- [ ] Public Key in GitHub eingefügt
- [ ] SSH Config erstellt
- [ ] Repository erfolgreich geklont
- [ ] Setup Script ausgeführt

**✅ Nach dieser Einrichtung funktioniert `git clone git@github.com:AlJ95/finetuning.git` problemlos!**
