.PHONY: fmt fmt-check test

fmt:
	go fmt ./...

fmt-check:
	test -z $$(gofmt -l .)

test:
	go test -v ./...