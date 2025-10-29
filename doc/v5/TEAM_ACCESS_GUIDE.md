# Team Access Management Guide

**Traffic Forecast v5.0 - Quick Team Setup**

This guide shows how to give your teammates full access to the GCP VM in under 2 minutes.

---

## Quick Start (< 2 Minutes)

### Option 1: One-Command Complete Setup (Recommended)

Use the Jupyter notebook for automated team access:

1. **Open the notebook:**

```bash
jupyter notebook notebooks/GCP_DEPLOYMENT.ipynb
```

2. **Scroll to "Team Access Management" section**

3. **Define your team:**

```python
my_team = [
{'name': 'alice', 'email': 'alice@gmail.com'},
{'name': 'bob', 'email': 'bob@gmail.com'},
]
```

4. **Run one command:**

```python
package_path = generate_team_access_package(my_team)
```

5. **What happens automatically:**

- Grants GCP project access to all emails
- Generates unique SSH keys for each person
- Creates connection scripts (Linux/Mac/Windows)
- Writes detailed README for each person
- Packages everything into one ZIP file

6. **Distribute:**
- Extract the ZIP file
- Send each teammate their folder (via email, Drive, etc.)
- They can connect immediately!

**Total time:** < 2 minutes to give full access to unlimited teammates!

---

## What Each Teammate Gets

When you run `generate_team_access_package()`, each teammate gets a folder with:

```
teammate_name/
├── id_rsa # Private SSH key (unique to them)
├── id_rsa.pub # Public SSH key
├── connect.sh # Linux/Mac connection script
├── connect.bat # Windows connection script
└── README.md # Complete instructions
```

### Teammate Usage (Super Easy!)

**Linux/Mac:**

```bash
chmod +x connect.sh
./connect.sh
```

**Windows:**

- Double-click `connect.bat`
- Or run in Git Bash: `./connect.sh`

**That's it!**They're connected to the VM.

---

## Manual Methods (If Needed)

### Method 1: Grant GCP Access Only

If teammates will use `gcloud` CLI instead of direct SSH:

```python
# In GCP_DEPLOYMENT.ipynb
grant_team_access(['alice@gmail.com', 'bob@gmail.com'])
```

**Teammates then:**

1. Install gcloud: https://cloud.google.com/sdk/docs/install
2. Login: `gcloud auth login`
3. Set project: `gcloud config set project YOUR_PROJECT_ID`
4. SSH: `gcloud compute ssh VM_NAME --zone=ZONE`

### Method 2: Setup SSH Key for One Person

```python
# In GCP_DEPLOYMENT.ipynb
setup_ssh_key_for_teammate('alice')
```

This generates SSH key and connection script for just one person.

---

## Security Best Practices

### 1. Unique Keys per Person

**DO:**

- Generate unique SSH key for each teammate
- Use `generate_team_access_package()` - it does this automatically

**DON'T:**

- Share your own SSH key
- Reuse the same key for multiple people

### 2. Secure Distribution

**DO:**

- Send via password-protected ZIP
- Use Google Drive with limited access
- Send via Signal/Telegram (encrypted messaging)

**DON'T:**

- Post keys on Slack/Discord
- Commit to GitHub
- Send via plain email (if possible)

### 3. Key Storage

**Tell your teammates:**

```bash
# Store key securely
chmod 600 id_rsa
mv id_rsa ~/.ssh/traffic-forecast-key

# Update connect script
ssh -i ~/.ssh/traffic-forecast-key user@VM_IP
```

---

## Managing Team Access

### View Current Access

```python
# In GCP_DEPLOYMENT.ipynb
list_team_members()
```

**Output:**

```
CURRENT TEAM ACCESS
======================================================================

Total users: 3

alice@gmail.com
• compute admin
• viewer

bob@gmail.com
• compute admin

SSH KEYS ON VM:
======================================================================
Total SSH keys: 2

alice
bob
```

### Revoke Access

When someone leaves the project:

```python
# In GCP_DEPLOYMENT.ipynb
revoke_team_access('alice@gmail.com', username='alice')
```

**This removes:**

- GCP project permissions
- SSH key from VM

**They can no longer:**

- Access GCP console
- SSH to VM
- View project resources

---

## Troubleshooting

### Teammate Can't Connect

**1. Check they have correct permissions:**

```python
list_team_members()
```

**2. Verify SSH key on VM:**

```bash
gcloud compute instances describe VM_NAME \
--zone=ZONE \
--format='get(metadata.items.ssh-keys)'
```

**3. Test from their machine:**

```bash
# Check SSH key permissions
ls -la id_rsa
# Should show: -rw------- (600)

# If not, fix it:
chmod 600 id_rsa

# Test connection
ssh -vvv -i id_rsa username@VM_IP
# -vvv shows detailed debug info
```

### "Permission denied (publickey)"

**Cause:**SSH key permissions too open

**Solution:**

```bash
chmod 600 id_rsa
```

### "Host key verification failed"

**Cause:**VM IP changed or first-time connection

**Solution:**

```bash
# Remove old host key
ssh-keygen -R VM_IP

# Or use -o StrictHostKeyChecking=no (less secure)
ssh -i id_rsa -o StrictHostKeyChecking=no user@VM_IP
```

### Teammate Already Has GCP Access But Can't SSH

**Cause:**SSH key not added to VM

**Solution:**

```python
# Regenerate SSH access
setup_ssh_key_for_teammate('username')
```

---

## Access Levels Comparison

### Full Access (Recommended for Team)

```python
grant_team_access('user@gmail.com', role='roles/compute.admin')
setup_ssh_key_for_teammate('username')
```

**Can:**

- SSH to VM
- Start/stop VM
- View logs
- Download data
- Modify VM settings
- Access GCP console

### Read-Only Access (For Observers)

```python
grant_team_access('observer@gmail.com', role='roles/viewer')
# Do NOT setup SSH key
```

**Can:**

- View GCP console
- See VM status
- View logs (via console)

**Cannot:**

- SSH to VM
- Start/stop VM
- Modify settings

### SSH-Only Access (No GCP Console)

```python
# Don't grant GCP access
setup_ssh_key_for_teammate('username')
```

**Can:**

- SSH to VM
- Run commands
- View data

**Cannot:**

- Access GCP console
- Start/stop VM (from console)
- View billing

---

## Common Team Scenarios

### Scenario 1: Academic Project (3-5 Students)

**Setup:**

```python
team = [
{'name': 'student1', 'email': 'student1@fpt.edu.vn'},
{'name': 'student2', 'email': 'student2@fpt.edu.vn'},
{'name': 'student3', 'email': 'student3@fpt.edu.vn'},
]

package = generate_team_access_package(team)
```

**Result:**

- Each student has full access
- Can all SSH to VM simultaneously
- Can monitor collection together
- Can download data independently

**Cost:** $0 additional (same VM serves all)

### Scenario 2: Mentor/Advisor Access

**Setup (Read-only):**

```python
grant_team_access('mentor@university.edu', role='roles/viewer')
```

**Result:**

- Mentor can view GCP console
- Can see VM status and logs
- Cannot make changes
- Cannot SSH

### Scenario 3: Temporary Collaborator

**Setup:**

```python
# Grant access
setup_ssh_key_for_teammate('collaborator')

# After project ends, revoke:
revoke_team_access('collab@gmail.com', username='collaborator')
```

### Scenario 4: Remote Team (Different Time Zones)

**Setup:**

```python
# All team members with full access
team = [
{'name': 'usa_teammate', 'email': 'usa@gmail.com'},
{'name': 'vietnam_teammate', 'email': 'vn@gmail.com'},
{'name': 'europe_teammate', 'email': 'eu@gmail.com'},
]

package = generate_team_access_package(team)
```

**Benefits:**

- Everyone can work independently
- No scheduling conflicts
- 24/7 monitoring possible
- Each person can download data in their timezone

---

## Team Access Checklist

Before sharing access, ensure:

- [ ] VM is running and tested
- [ ] Collection is working (100% success rate)
- [ ] You have teammates' Gmail addresses
- [ ] You've chosen appropriate access level (full/read-only)
- [ ] You have a secure way to distribute keys
- [ ] Teammates know the project details (VM IP, project ID, etc.)

After sharing access:

- [ ] Verify each teammate can connect
- [ ] Show them where logs are (`~/traffic-forecast/logs/`)
- [ ] Show them how to download data
- [ ] Share this guide with them
- [ ] Set up a team communication channel (Slack, Discord, etc.)

---

## Quick Reference

### All-in-One Team Setup

```python
# Define team
team = [
{'name': 'alice', 'email': 'alice@gmail.com'},
{'name': 'bob', 'email': 'bob@gmail.com'},
]

# Generate complete package
package = generate_team_access_package(team)

# Extract and send each person their folder
# Total time: < 2 minutes!
```

### Individual Access

```python
# Grant GCP access
grant_team_access('user@gmail.com')

# Setup SSH
setup_ssh_key_for_teammate('username')
```

### Management

```python
# View access
list_team_members()

# Revoke access
revoke_team_access('user@gmail.com', username='username')
```

---

## Pro Tips

1. **Use descriptive usernames:**

```python
# Good
setup_ssh_key_for_teammate('alice_frontend')
setup_ssh_key_for_teammate('bob_ml')

# Bad
setup_ssh_key_for_teammate('user1')
setup_ssh_key_for_teammate('temp')
```

2. **Generate package once, share multiple times:**

```python
# Generate once
package = generate_team_access_package(team)

# Keep the ZIP safe
# Send individual folders as people join
```

3. **Document your team in code:**

```python
# Keep this in your notebook for reference
TEAM_ROSTER = {
'alice': {
'email': 'alice@gmail.com',
'role': 'ML Engineer',
'access_granted': '2025-01-28',
},
'bob': {
'email': 'bob@gmail.com',
'role': 'Data Analyst',
'access_granted': '2025-01-28',
}
}
```

4. **Automate revocation on project end:**
```python
# At end of project, revoke all at once
for name, info in TEAM_ROSTER.items():
revoke_team_access(info['email'], username=name)
```

---

## Support

**For access issues:**

1. Check this guide first
2. Try troubleshooting section
3. Run `list_team_members()` to verify access
4. Contact project admin

**Common questions:**

- **Q:**Can teammates see each other's data?
**A:**Yes, if they have SSH access to VM, they can see all data in `~/traffic-forecast/data/`

- **Q:**Can multiple teammates SSH at once?
**A:**Yes! Linux supports multiple concurrent SSH sessions.

- **Q:**Does each teammate cost more?
**A:**No! They share the same VM, no additional compute cost.

- **Q:**What if I forget to revoke access?
**A:**Access persists until manually revoked. Use `revoke_team_access()` when needed.

---

## Success Stories

**Before (Manual Setup):**

- 30 minutes per teammate
- Error-prone (typos in permissions)
- Multiple back-and-forth emails
- SSH key permission issues

**After (generate_team_access_package):**

- < 2 minutes for unlimited teammates
- Automated, zero errors
- One ZIP file with everything
- Teammates connect on first try

---

**Version:** 5.0.0
**Last updated:**January 2025
**Related:** `notebooks/GCP_DEPLOYMENT.ipynb` (Team Access Management section)
