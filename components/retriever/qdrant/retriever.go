package qdrant

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/qdrant/go-client/qdrant"
)

type RetrieverConfig struct {
	Client         *qdrant.Client
	Collection     string
	Embedding      embedding.Embedder
	TopK           int
	ScoreThreshold float32
}

type Retriever struct {
	config *RetrieverConfig
}

func NewRetriever(ctx context.Context, config *RetrieverConfig) (*Retriever, error) {
	if config.Client == nil {
		return nil, fmt.Errorf("[QdrantRetriever] client not provided")
	}
	if config.Collection == "" {
		config.Collection = "eino_collection"
	}
	if config.TopK == 0 {
		config.TopK = 5
	}
	if config.Embedding == nil {
		return nil, fmt.Errorf("[QdrantRetriever] embedding not provided")
	}
	return &Retriever{config: config}, nil
}

func defaultPayloadToDocument(ctx context.Context, point *qdrant.ScoredPoint) (*schema.Document, error) {
	payload := point.GetPayload()
	id := payload["id"].GetStringValue()
	content := payload["content"].GetStringValue()
	meta := map[string]interface{}{}
	for k, v := range payload {
		if k != "id" && k != "content" {
			meta[k] = v
		}
	}
	return &schema.Document{
		ID:       id,
		Content:  content,
		MetaData: meta,
	}, nil
}

func (r *Retriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	vecs, err := r.config.Embedding.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("[QdrantRetriever] embedding failed: %w", err)
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("[QdrantRetriever] no embedding returned")
	}
	vector := make([]float32, len(vecs[0]))
	for i, v := range vecs[0] {
		vector[i] = float32(v)
	}
	searchRes, err := r.config.Client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: r.config.Collection,
		Query:          qdrant.NewQuery(vector...),
		Limit:          qdrant.PtrOf(uint64(r.config.TopK)),
	})
	if err != nil {
		return nil, fmt.Errorf("[QdrantRetriever] search failed: %w", err)
	}
	var docs []*schema.Document
	for _, pt := range searchRes {
		if r.config.ScoreThreshold > 0 && pt.Score < r.config.ScoreThreshold {
			continue
		}
		doc, err := defaultPayloadToDocument(ctx, pt)
		if err != nil {
			return nil, fmt.Errorf("[QdrantRetriever] payload conversion failed: %w", err)
		}
		docs = append(docs, doc)
	}
	return docs, nil
}
