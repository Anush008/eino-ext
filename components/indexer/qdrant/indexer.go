package qdrant

import (
	context "context"
	fmt "fmt"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/schema"
	"github.com/qdrant/go-client/qdrant"
)

type IndexerConfig struct {
	// Qdrant client
	Client *qdrant.Client
	// Collection name
	Collection string
	// Embedding vectorization method
	Embedding embedding.Embedder
	// BatchSize for upserts
	BatchSize int
}

type Indexer struct {
	config *IndexerConfig
}

func NewIndexer(ctx context.Context, config *IndexerConfig) (*Indexer, error) {
	if config.Client == nil {
		return nil, fmt.Errorf("[NewIndexer] qdrant client not provided")
	}
	if config.Embedding == nil {
		return nil, fmt.Errorf("[NewIndexer] embedding not provided")
	}
	if config.Collection == "" {
		config.Collection = "eino_collection"
	}
	if config.BatchSize == 0 {
		config.BatchSize = 10
	}
	if config.DocumentToPayload == nil {
		config.DocumentToPayload = defaultDocumentToPayload
	}
	return &Indexer{config: config}, nil
}

func defaultDocumentToPayload(ctx context.Context, doc *schema.Document) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"id":      doc.ID,
		"content": doc.Content,
	}
	for k, v := range doc.MetaData {
		payload[k] = v
	}
	return payload, nil
}

func (i *Indexer) Store(ctx context.Context, docs []*schema.Document, opts ...interface{}) (ids []string, err error) {
	if len(docs) == 0 {
		return nil, nil
	}
	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.Content)
	}
	vectors, err := i.config.Embedding.EmbedStrings(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("[Qdrant.Store] embedding failed: %w", err)
	}
	if len(vectors) != len(docs) {
		return nil, fmt.Errorf("[Qdrant.Store] embedding result length mismatch: want %d, got %d", len(docs), len(vectors))
	}
	points := make([]*qdrant.PointStruct, 0, len(docs))
	for idx, doc := range docs {
		payload, err := i.config.DocumentToPayload(ctx, doc)
		if err != nil {
			return nil, fmt.Errorf("[Qdrant.Store] payload conversion failed: %w", err)
		}
		vec := make([]float32, len(vectors[idx]))
		for j, v := range vectors[idx] {
			vec[j] = float32(v)
		}
		points = append(points, &qdrant.PointStruct{
			Id:      &qdrant.PointId{PointIdOptions: &qdrant.PointId_Uuid{Uuid: doc.ID}},
			Vectors: &qdrant.Vectors{VectorsOptions: &qdrant.Vectors_Vector{Vector: &qdrant.Vector{Data: vec}}},
			Payload: qdrant.NewValueMap(payload),
		})
	}
	batchSize := i.config.BatchSize
	if batchSize <= 0 {
		batchSize = 10
	}
	for start := 0; start < len(points); start += batchSize {
		end := start + batchSize
		if end > len(points) {
			end = len(points)
		}
		_, err := i.config.Client.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: i.config.Collection,
			Points:         points[start:end],
		})
		if err != nil {
			return nil, fmt.Errorf("[Qdrant.Store] Upsert failed: %w", err)
		}
	}
	ids = make([]string, len(docs))
	for idx, doc := range docs {
		ids[idx] = doc.ID
	}
	return ids, nil
}

const typ = "Qdrant"

func (i *Indexer) GetType() string {
	return typ
}

func (i *Indexer) IsCallbacksEnabled() bool {
	return true
}
