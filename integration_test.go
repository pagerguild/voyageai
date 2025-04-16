//go:build integration
// +build integration

package voyageai

import (
	"reflect"
	"testing"
)

func TestEmbeddingWithDifferentEncodings(t *testing.T) {
	// Create client using environment variable
	client := NewClient(nil)

	// Test texts to embed
	texts := []string{"This is a test sentence for embedding."}
	model := "voyage-3-large"

	// First request with default encoding
	defaultResp, err := client.Embed(texts, model, nil)
	if err != nil {
		t.Fatalf("Failed to get embeddings with default encoding: %v", err)
	}

	// Second request with base64 encoding
	base64Resp, err := client.Embed(texts, model, &EmbeddingRequestOpts{
		EncodingFormat: Opt("base64"),
	})
	if err != nil {
		t.Fatalf("Failed to get embeddings with base64 encoding: %v", err)
	}

	// Verify both responses have the same model
	if defaultResp.Model != base64Resp.Model {
		t.Errorf("Model mismatch: default=%s, base64=%s", defaultResp.Model, base64Resp.Model)
	}

	// Verify both responses have the same number of embeddings
	if len(defaultResp.Data) != len(base64Resp.Data) {
		t.Errorf("Embedding count mismatch: default=%d, base64=%d",
			len(defaultResp.Data), len(base64Resp.Data))
	}

	// Verify embeddings are equivalent
	if len(defaultResp.Data) > 0 && len(base64Resp.Data) > 0 {
		defaultEmbedding := defaultResp.Data[0].Embedding
		base64Embedding := base64Resp.Data[0].Embedding

		base64Slice, err := base64Embedding.ToSlice()
		if err != nil {
			t.Errorf("Failed to decode base64 embedding: %v", err)
		}

		if len(defaultEmbedding.AsNumeric) != len(base64Slice) {
			t.Errorf("Embedding dimension mismatch: default=%d, base64=%d",
				len(defaultEmbedding.AsNumeric), len(base64Slice))
		} else {
			// Compare actual values (should be the same regardless of encoding format)
			for i := range defaultEmbedding.AsNumeric {
				if !almostEqual(defaultEmbedding.AsNumeric[i], base64Slice[i], 1e-6) {
					t.Errorf("Embedding value mismatch at index %d: default=%f, base64=%f",
						i, defaultEmbedding.AsNumeric[i], base64Slice[i])
				}
			}
		}
	}

	// Make sure the responses are equivalent when comparing their data structures
	// (excluding the actual embedding values which we already compared)
	defaultRespCopy := *defaultResp
	base64RespCopy := *base64Resp

	// Set embeddings to nil so we can compare the rest of the structure
	if len(defaultRespCopy.Data) > 0 {
		defaultRespCopy.Data[0].Embedding = Embedding[float32]{}
	}
	if len(base64RespCopy.Data) > 0 {
		base64RespCopy.Data[0].Embedding = Embedding[float32]{}
	}

	if !reflect.DeepEqual(defaultRespCopy, base64RespCopy) {
		t.Errorf("Response structure mismatch after excluding embeddings")
	}
}

func TestEmbedInt8WithDifferentEncodings(t *testing.T) {
	// Create client using environment variable
	client := NewClient(nil)

	// Test texts to embed
	texts := []string{"This is a test sentence for int8 embedding."}
	model := "voyage-3-large"

	// First request with default encoding
	defaultResp, err := client.EmbedInt8(texts, model, &EmbeddingRequestOpts{
		OutputDType: Opt("int8"),
	})
	if err != nil {
		t.Fatalf("Failed to get int8 embeddings with default encoding: %v", err)
	}

	// Second request with base64 encoding
	base64Resp, err := client.EmbedInt8(texts, model, &EmbeddingRequestOpts{
		OutputDType:    Opt("int8"),
		EncodingFormat: Opt("base64"),
	})
	if err != nil {
		t.Fatalf("Failed to get int8 embeddings with base64 encoding: %v", err)
	}

	// Verify both responses have the same model
	if defaultResp.Model != base64Resp.Model {
		t.Errorf("Model mismatch: default=%s, base64=%s", defaultResp.Model, base64Resp.Model)
	}

	// Verify both responses have the same number of embeddings
	if len(defaultResp.Data) != len(base64Resp.Data) {
		t.Errorf("Embedding count mismatch: default=%d, base64=%d",
			len(defaultResp.Data), len(base64Resp.Data))
	}

	// Verify embeddings are equivalent
	if len(defaultResp.Data) > 0 && len(base64Resp.Data) > 0 {
		defaultEmbedding := defaultResp.Data[0].Embedding
		base64Embedding := base64Resp.Data[0].Embedding

		base64Slice, err := base64Embedding.ToSlice()
		if err != nil {
			t.Errorf("Failed to decode base64 embedding: %v", err)
		}

		if len(defaultEmbedding.AsNumeric) != len(base64Slice) {
			t.Errorf("Embedding dimension mismatch: default=%d, base64=%d",
				len(defaultEmbedding.AsNumeric), len(base64Slice))
		} else {
			// Compare actual values (should be the same regardless of encoding format)
			for i := range defaultEmbedding.AsNumeric {
				if defaultEmbedding.AsNumeric[i] != base64Slice[i] {
					t.Errorf("Embedding value mismatch at index %d: default=%d, base64=%d",
						i, defaultEmbedding.AsNumeric[i], base64Slice[i])
				}
			}
		}
	}

	// Make sure the responses are equivalent when comparing their data structures
	// (excluding the actual embedding values which we already compared)
	defaultRespCopy := *defaultResp
	base64RespCopy := *base64Resp

	// Set embeddings to nil so we can compare the rest of the structure
	if len(defaultRespCopy.Data) > 0 {
		defaultRespCopy.Data[0].Embedding = Embedding[int8]{}
	}
	if len(base64RespCopy.Data) > 0 {
		base64RespCopy.Data[0].Embedding = Embedding[int8]{}
	}

	if !reflect.DeepEqual(defaultRespCopy, base64RespCopy) {
		t.Errorf("Response structure mismatch after excluding embeddings")
	}
}

func TestEmbedUint8WithDifferentEncodings(t *testing.T) {
	// Create client using environment variable
	client := NewClient(nil)

	// Test texts to embed
	texts := []string{"This is a test sentence for uint8 embedding."}
	model := "voyage-3-large"

	// First request with default encoding
	defaultResp, err := client.EmbedUint8(texts, model, &EmbeddingRequestOpts{
		OutputDType: Opt("uint8"),
	})
	if err != nil {
		t.Fatalf("Failed to get uint8 embeddings with default encoding: %v", err)
	}

	// Second request with base64 encoding
	base64Resp, err := client.EmbedUint8(texts, model, &EmbeddingRequestOpts{
		OutputDType:    Opt("uint8"),
		EncodingFormat: Opt("base64"),
	})
	if err != nil {
		t.Fatalf("Failed to get uint8 embeddings with base64 encoding: %v", err)
	}

	// Verify both responses have the same model
	if defaultResp.Model != base64Resp.Model {
		t.Errorf("Model mismatch: default=%s, base64=%s", defaultResp.Model, base64Resp.Model)
	}

	// Verify both responses have the same number of embeddings
	if len(defaultResp.Data) != len(base64Resp.Data) {
		t.Errorf("Embedding count mismatch: default=%d, base64=%d",
			len(defaultResp.Data), len(base64Resp.Data))
	}

	// Verify embeddings are equivalent
	if len(defaultResp.Data) > 0 && len(base64Resp.Data) > 0 {
		defaultEmbedding := defaultResp.Data[0].Embedding
		base64Embedding := base64Resp.Data[0].Embedding

		base64Slice, err := base64Embedding.ToSlice()
		if err != nil {
			t.Errorf("Failed to decode base64 embedding: %v", err)
		}

		if len(defaultEmbedding.AsNumeric) != len(base64Slice) {
			t.Errorf("Embedding dimension mismatch: default=%d, base64=%d",
				len(defaultEmbedding.AsNumeric), len(base64Slice))
		} else {
			// Compare actual values (should be the same regardless of encoding format)
			for i := range defaultEmbedding.AsNumeric {
				if defaultEmbedding.AsNumeric[i] != base64Slice[i] {
					t.Errorf("Embedding value mismatch at index %d: default=%d, base64=%d",
						i, defaultEmbedding.AsNumeric[i], base64Slice[i])
				}
			}
		}
	}

	// Make sure the responses are equivalent when comparing their data structures
	// (excluding the actual embedding values which we already compared)
	defaultRespCopy := *defaultResp
	base64RespCopy := *base64Resp

	// Set embeddings to nil so we can compare the rest of the structure
	if len(defaultRespCopy.Data) > 0 {
		defaultRespCopy.Data[0].Embedding = Embedding[uint8]{}
	}
	if len(base64RespCopy.Data) > 0 {
		base64RespCopy.Data[0].Embedding = Embedding[uint8]{}
	}

	if !reflect.DeepEqual(defaultRespCopy, base64RespCopy) {
		t.Errorf("Response structure mismatch after excluding embeddings")
	}
}

// Helper function to compare float values with tolerance
func almostEqual(a, b float32, tolerance float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tolerance
}
