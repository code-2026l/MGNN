#!/bin/bash
# Install git-lfs (corrected: tar dir has no 'v' prefix).
set -euo pipefail
cd /tmp
rm -rf git-lfs-3.5.1 ~/bin
mkdir -p ~/bin
tar xzf gitlfs.tar.gz
cp git-lfs-3.5.1/git-lfs ~/bin/
chmod +x ~/bin/git-lfs
export PATH="$HOME/bin:$PATH"
~/bin/git-lfs version 2>&1 | head -1