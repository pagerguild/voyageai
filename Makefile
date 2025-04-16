.PHONY: fmt fmt-check test integration_test

fmt:
	go fmt ./...

fmt-check:
	test -z $$(gofmt -l .)

test:
	go test -v ./...

integration_test:
	go test -v -tags=integration ./...
