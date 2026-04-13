import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(catalog, users) {
    // Validate inputs
    if (!users || !Array.isArray(users)) {
        console.error('Invalid users data:', users);
        throw new Error('Users must be a non-empty array');
    }
    
    if (!catalog || !Array.isArray(catalog)) {
        console.error('Invalid catalog data:', catalog);
        throw new Error('Catalog must be a non-empty array');
    }

    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

    const colorIndex = colors.map((color, index) => [color, index]);

    const categoriesIndex = categories.map((category, index) => [category, index]); 

    // Computar a média de idade dos compradores por produto
    // (ajuda a personalizar as recomendações com base na idade)
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        if (user.purchasedProducts && Array.isArray(user.purchasedProducts)) {
            user.purchasedProducts.forEach(p => {
                ageSums[p] = (ageSums[p] || 0) + user.age;
                ageCounts[p] = (ageCounts[p] || 0) + 1; 
            })
        }
    })

    const productAvgAgeNorm = Object.fromEntries(
        catalog.map(p => {
            const avg = ageCounts[p.name] ? ageSums[p.name] / ageCounts[p.name] : midAge;
            return [p.name, normalize(avg, minAge, maxAge)];
        })
    );
    
    return {
        catalog,
        users,
        colorIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        dimentions: 2 + categories.length + colors.length, // idade + preço + categorias + cores
        productAvgAgeNorm,
    };
}

function encodeProduct(product, context) {
    // normalizando dados para ficar de 0 a 1
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price]);
    const age = tf.tensor1d([context.productAvgAgeNorm[product.name] ?? 0.5 * WEIGHTS.age]);
    
    const categoryEntry = context.categoriesIndex.find(([cat]) => cat === product.category);
    const categoryIndex = categoryEntry ? categoryEntry[1] : 0;
    const category = tf.oneHot(categoryIndex, context.numCategories).mul(WEIGHTS.category);
    
    const colorEntry = context.colorIndex.find(([col]) => col === product.color);
    const colorIndex = colorEntry ? colorEntry[1] : 0;
    const color = tf.oneHot(colorIndex, context.numColors).mul(WEIGHTS.color);
    
    return tf.concat1d([price, age, category, color]);
}

async function trainModel({ users }) {
    try {
        console.log('Training model with users:', users)
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
        const catalog = await (await fetch('/data/products.json')).json();
        
        const context = makeContext(catalog, users)

        context.productVectors = catalog.map(product => {
            return {
                name: product.name,
                meta: {...product},
                vector: encodeProduct(product, context).dataSync(), // Convertendo o tensor para array normal para facilitar o uso posterior
            }
        });
        
        _globalCtx = context;

        postMessage({
            type: workerEvents.trainingLog,
            epoch: 1,
            loss: 1,
            accuracy: 1
        });

        setTimeout(() => {
            postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
            postMessage({ type: workerEvents.trainingComplete });
        }, 1000);
    } catch (error) {
        console.error('Error training model:', error);
        postMessage({ 
            type: workerEvents.trainingError, 
            error: error.message 
        });
    }
}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
