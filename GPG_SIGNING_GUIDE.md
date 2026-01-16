# GPG 서명 가이드 (GPG Signing Guide)

**작성 일자**: 2026-01-17  
**목적**: Git 커밋 및 태그에 GPG 서명을 추가하여 무결성 보장

---

## 🎯 목적

GPG 서명을 통해:
- 코드 무결성 보장
- 작성자 인증
- 블록체인 해시 기록 강화
- 공개 발매 증명

---

## 🔑 GPG 키 생성

### 1. GPG 키 생성
```bash
gpg --full-generate-key
```

### 2. 키 타입 선택
- RSA and RSA (기본값)
- 키 크기: 4096
- 만료: 1년 (또는 원하는 기간)

### 3. 사용자 정보 입력
- 이름: [이름]
- 이메일: [GitHub 이메일]
- 코멘트: [선택사항]

---

## 🔐 Git에 GPG 키 설정

### 1. GPG 키 ID 확인
```bash
gpg --list-secret-keys --keyid-format LONG
```

### 2. Git에 GPG 키 설정
```bash
git config --global user.signingkey [GPG_KEY_ID]
git config --global commit.gpgsign true
```

---

## 📝 커밋 서명

### 자동 서명 (권장)
```bash
# 모든 커밋 자동 서명
git config --global commit.gpgsign true
```

### 수동 서명
```bash
git commit -S -m "커밋 메시지"
```

---

## 🏷️ 태그 서명

### 태그 생성 및 서명
```bash
git tag -s v1.0.0 -m "Neurons Engine v1.0.0 - Public Release"
```

### 태그 푸시
```bash
git push origin v1.0.0
```

### 태그 서명 확인
```bash
git tag -v v1.0.0
```

---

## 🔍 서명 확인

### 커밋 서명 확인
```bash
git log --show-signature
```

### 태그 서명 확인
```bash
git tag -v v1.0.0
```

---

## 📋 GitHub에 GPG 키 등록

### 1. 공개 키 내보내기
```bash
gpg --armor --export [GPG_KEY_ID]
```

### 2. GitHub에 등록
1. GitHub → Settings → SSH and GPG keys
2. New GPG key 클릭
3. 공개 키 붙여넣기
4. Add GPG key 클릭

---

## ✅ 확인 사항

- [ ] GPG 키 생성 완료
- [ ] Git에 GPG 키 설정 완료
- [ ] GitHub에 GPG 키 등록 완료
- [ ] 커밋 서명 테스트 완료
- [ ] 태그 서명 테스트 완료

---

## 🔗 참고 자료

- [Git GPG 서명 가이드](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [GitHub GPG 키 설정](https://docs.github.com/en/authentication/managing-commit-signature-verification)

---

**Last Updated**: 2026-01-17  
**Version**: v1.0.0

