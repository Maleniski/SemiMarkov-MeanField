import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from PIL import Image
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")


with st.sidebar:
    st.image("logo-unison.png", width=300)  # Adjust path and width as needed
    st.markdown("""
    Universidad de Sonora   
    Programa de Doctorado en Ciencias Matemáticas  
    ---------------
        
    **Elaborado por:** María Elena Martínez Manzanares  
    **Nombre de tesis:** Modelos de Control semi-Markovianos para sistemas de grandes poblaciones bajo un enfoque de campo medio   
    **Director de tesis:** Dr. Jesús Adolfo Minjárez Sosa  
    **Fecha de última modificación:** 01-12-2023  
    *Agradecimientos a la Dra. Carmen Geraldi Higuera Chan por sus valiosas sugerencias en la realización de este tablero.*
    """)

#region Cabecera

st.markdown("""# Simulación de la evolución de proporciones de objetos en una cantidad finita de clases bajo un enfoque de campo medio""")


st.markdown("""
            
            En este tablero podrá simular la evolución de las proporciones de objetos distribuidos en una cantidad finita de clases. 
            Cuando usted accede por primera vez, la simulación es generada con valores por defecto, pero es posible
            personalizar esto. Para poder realizar su simulación con los valores de su preferencia, es necesario indicar lo siguiente:

            - Número de iteraciones (o etapas) que desea simular. 
            - Cantidad de objectos en cada clase y matriz de transición entre clases, la cual debe subir en un único csv.
            """)

with st.expander("Teoría detrás del tablero"):
    st.markdown(
        """
        Dado un sistema controlado que se compone de una cantidad finita de clases $S=\{1,2,...,s\}$ y $N$ objetos que se distribuyen a lo largo del conjunto
        de $S$ clases, es posible estudiar la evolución del sistema en las etapas por medio de la proporción de objetos de cada clase.  

        Particularmente, si denotamos a $\{M^N(k)\}_{k\in\mathbb{N}_0}$ los vectores de proporciones de los objetos en las $S$ clases durante las épocas
        $k=0,1,2...$, es posible demostrar que el proceso $\{M^N(k)\}_{k\in\mathbb{N}_0}$ es una cadena de Markov. Este resultado nos permite utilizar 
        métodos de simulación de Monte Carlo para obtener una ecuación en diferencias estocásticas que define la evolución del proceso. 
        Específicamente obtenemos 
        """)
    st.latex(r"""M^N(k+1)=H^N(M^N(k),a_k,w_k), k\in\mathbb{N}_0""")
    st.markdown("""
        donde $\{w_k\}$ es una sucesión de variables aleatorias i.i.d. en $\mathbb{R}^N$ con distribución theta (o uniformes).  

        Para lograr esto, comencemos considerando $a\in A$, $i,j\in S$, y la siguiente partición del $[0,1]$
        $$
        \Delta_{ij}(a):=[\phi_{i(j-1)}(a),\phi_{ij}(a)]\subseteq [0,1], 
        $$
        donde 
        $$
        \phi_{i0}(a)\equiv 0,\ \ \phi_{ij}(a):=\sum_{l=1}^{j}K_{il}(a),i,j\in S.
        $$
        Para cada $i\in S$, y $k\in\mathbb{N}_0$, definimos
        $$
        w^{i}(k):=(w_{1}^{i}(k),...,w_{NM_{i}^{N}(k)}^{i}(k));
        $$  
        $$
        w_{k}:=\{w^{i}(k)\}_{i\in S}.
        $$
        donde $\{w_n^i(k)\}$  es una familia de variables aleatorias uniformemente distribuidas en $[0,1]$ con $n\in\{1,2,...,N\}$. Considerando que $\sum_{j=1}^{s} NM_j^N(k)=N$, se tiene que $w_k\in[0,1]^N$. Entonces
        """)
    st.latex(r"""M_{j}^{N}(k+1)= \frac{1}{N}\sum_{i=1}^{s}\sum_{n=1}^{NM_{i}^{N}(k)}1_{\Delta_{ij}(a_{k})}(w_{n}^{i}(k)). \quad (1)""")
    st.markdown("""
        Finalmente, se define la función $H^N$ como
        """)
    st.latex(r"""H^{N}(m,a,w):=\{H_{i}^{N}(m,a,w)\}_{i\in S},\quad(m,a,w)\in\mathbb{P}_{N}(S) \times A \times [0,1]^{N},""")
    st.markdown("""
        donde
        """)
    st.latex(r"""H_{j}^{N}(m,a,w)=\frac{1}{N}\sum_{i=1}^{s}\sum_{n=1}^{Nm_{i}}1_{\Delta _{ij}(a)}(w_{n}^{i}(k)),""")
    st.markdown("""    
        y $m=\{m_{i}\}_{i\in S}$, $k\in \mathbb{N}_0$.  

        En este tablero, por simpleza consideraremos evoluciones no controladas, es decir, de la forma $M^N(k+1)=H^N(M^N(k),w_k)$ con $K_{ij}(a)\equiv K_{ij}$.  

        Una característica que puede ser demostrada y cumplen las proporciones determinadas por (1), es que conforme el número de objetos $N$ crece, 
        la distribución (o también llamada configuración) de los objetos converge a una configuración límite. Este resultado se le conoce actualmente 
        en control estocástico como *convergencia de campo medio*. 

        La configuración límite, o de campo medio, se parecerá a la configuración original del sistema, y viene dada por la función
    """
    )
    st.latex(r"""H(m)=mK \quad (2)""")
    st.markdown("""
        en donde $m$ es un vector de proporción entre clases y $K := [K_{ij}]$. Es decir, (2) es el producto matricial de un vector con una matriz.
                
        Por medio de este tablero, en la gráfica izquierda es posible ver la evolución de la configuración de objetos simulada por medio de (1). 
        Cuando $N$ es suficientemente grande, los valores que se observan graficados puede ser considerados como los valores del campo medio.  

        Adicionalmente, en la gráfica de la derecha es posible la configuración de objetos simulados por medio de (1), comparada con los resultados de la simulación
        de (2). Es decir, se contrasta la simulación del proceso original con la simulación de campo medio, para ver que estas son en efecto similares.
        
        En este tablero, dado que python es cero indexado, la enumeración en $S$ comienza en $0$, es decir, $S=\{0,1,2,...,s-1\}$.
    """
    )
#endregion

#region Inicialización default del tablero e inputs de usuario

# Define default transition matrix
default_transition_matrix = np.array([
    [0.8, 0.15, 0.05],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6]
])
num_states = default_transition_matrix.shape[0]

# User input for the number of iterations
num_iterations = st.number_input('Número de iteraciones', value=10, min_value=1)
 
# Set up the file uploader
uploaded_file = st.file_uploader("Cargue la matriz con el número de objetos en cada clase en el primer renglón, y la matriz de transición como CSV", type=["csv"])

st.markdown("""
            Una matriz con formato correcto para este tablero es de la siguiente forma. En el primer renglón está la cantidad de objetos en cada clase, y
            del segundo renglón en adelante, se encuentra la matriz de transición. Se presenta un ejemplo a continuación:
            
            |     |      |      |
            | --- | ---  | ---  |
            | 50  | 75   | 30   |
            | 0.8 | 0.15 | 0.05 |
            | 0.3 | 0.4  | 0.3  |
            | 0.1 | 0.3  | 0.6  |
            
            """)


# Initialize with default values if no file is uploaded
transition_matrix = default_transition_matrix if uploaded_file is None else None
initial_population = [50, 30, 20] if uploaded_file is None else None
population_size = np.sum(initial_population) if uploaded_file is None else None
default_proportions = initial_population/population_size if uploaded_file is None else None


if uploaded_file is not None:
    # Read the CSV into a DataFrame and convert to a NumPy array
    try:
        data_df = pd.read_csv(uploaded_file, header=None)
        # Extract the initial population from the first row
        initial_population = data_df.iloc[0].astype(int).values
        # Extract the transition matrix from the remaining rows
        transition_matrix = data_df.iloc[1:].values
        num_states = transition_matrix.shape[0]
        population_size = initial_population.sum()  # Update population size based on uploaded data

        # Display current transition matrix and population
        st.write("Cantidad de objetos en cada clase (primer renglón):")
        st.write(initial_population)
        st.write("Matriz de transición (del segundo renglón en adelante):")
        st.write(transition_matrix)

    except Exception as e:
        st.error(f"Se produjo un error al leer el archivo CSV: {e}")
        st.stop()


submit_button = st.button('Iniciar simulación')


#endregion

#region Definición de funciones de simulación

# Function to simulate the Markov chain for a population
def simulate_markov_chain_population(num_states, initial_population, transition_matrix, num_iterations):
    current_population = initial_population
    population_history = [current_population]

    for _ in range(num_iterations):
        new_population = np.zeros(num_states, dtype=int)

        for i, state_count in enumerate(current_population):
            cumulative_prob = np.cumsum(transition_matrix[i])

            for _ in range(state_count):
                random_value = np.random.uniform(0, 1)
                next_state_idx = np.searchsorted(cumulative_prob, random_value)
                new_population[next_state_idx] += 1

        population_history.append(new_population)
        current_population = new_population

    return population_history

def mean_field_configuration(proportion, transition_matrix):
    return np.dot(proportion,transition_matrix)

def generate_colors(num_classes):
    # Generate a colormap and pick colors from it
    colormap = plt.cm.get_cmap('tab10', num_classes)  # 'tab10' is a good colormap with distinct colors
    return [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, ' for r, g, b, _ in [colormap(i) for i in range(num_classes)]]

#endregion

#region Gráficas

# Run the simulation with the given parameters
simulated_population_history = simulate_markov_chain_population(num_states, initial_population, transition_matrix, num_iterations)

# Calculate the proportion vector for each iteration
proportion_vectors = [pop / population_size for pop in simulated_population_history]

if uploaded_file is None:
    colors = ['rgba(31, 119, 180,', 'rgba(255, 127, 14,', 'rgba(44, 160, 44,']
else:
    colors = generate_colors(num_states)

alpha_original = '1)'
alpha_mean_field = '0.6)' 

# Columns for layout
col1, col2 = st.columns((1, 1))

with col1:
    fig = go.Figure()

    # Iterate over each class and add a bar to the stack
    for i in range(num_states):
        fig.add_trace(go.Bar(
            x=list(range(num_iterations + 1)),
            y=[pv[i] for pv in proportion_vectors],
            name=f'Clase {i}',
            marker_color=colors[i] + alpha_original
        ))

    # Update the layout of the plot to stack the bars
    fig.update_layout(
        barmode='stack',
        title=f'Evolución de configuración del sistema de {population_size} objetos en las etapas',
        xaxis_title='Número de iteración',
        yaxis_title='Proporción',
        legend_title='Clases'
    )

    # Display the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    st.write("Tabla de configuraciones por iteración")
    df_proportion_vectors = pd.DataFrame(proportion_vectors, columns=[f'Clase {i}' for i in range(num_states)])
    df_proportion_vectors.index.rename("Número de iteración", inplace=True)
    st.dataframe(df_proportion_vectors)    

    st.markdown("""
        ---
                
        **Explicación:**  
                
        En la gráfica superior podemos ver la evolución de las configuraciones en las etapas $k=0,1,2,...,$ considerando una cantidad $N$ de objetos fija.
        Cuando $N$ es muy grande, podemos considerar que los valores que se observan graficados son los valores que se tendríamos en el proceso de campo medio.  
    """)


with col2:
    # Simulate mean field configuration evolution
    mean_field_proportion_vectors = [proportion_vectors[0]]
    for _ in range(1, num_iterations + 1):
        new_proportions = mean_field_configuration(mean_field_proportion_vectors[-1], transition_matrix)
        mean_field_proportion_vectors.append(new_proportions)

    # Initialize a figure for stacked bar chart
    fig = go.Figure()

    # Add original and mean field configurations as side-by-side stacked bars for each iteration
    for i in range(num_states):
        # Original evolution bars
        fig.add_trace(go.Bar(
            name=f'Modelo Original - Clase {i}',
            x=[(x*2) for x in range(num_iterations + 1)],  # Even number positions for original
            y=[pv[i] for pv in proportion_vectors],
            offsetgroup=i,  # Assign an offsetgroup for each class
            marker_color=colors[i] + alpha_original,
        ))

        # Mean field configuration bars
        fig.add_trace(go.Bar(
            name=f'Campo Medio - Clase {i}',
            x=[(x*2 + 0.4) for x in range(num_iterations + 1)],  # Slightly offset within the group for mean field
            y=[pv[i] for pv in mean_field_proportion_vectors],
            offsetgroup=i,  # Use the same offsetgroup for mean field
            marker_color=colors[i] + alpha_mean_field,
        ))

    # Update the layout of the plot
    fig.update_layout(
        barmode='stack',
        title='Comparación de evolución de la configuración original y configuración de Campo Medio',
        xaxis=dict(
            title='Número de iteración',
            tickmode='array',
            tickvals=[x*2 + 0.2 for x in range(num_iterations + 1)],  # Center ticks in the group
            ticktext=[str(x) for x in range(num_iterations + 1)],
        ),
        yaxis_title='Proporción',
        legend_title='Modelos y clases',
        bargroupgap=0,  # Set a gap between groups of bars
        bargap=0.0001,  # Reduce the gap between bars within a group for wider bars
    )


    # Display the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    st.write("Tabla de Configuración de Campo Medio")

    # Since the mean field proportions are constant, we take the last calculated mean field proportions

    df_mean_field = pd.DataFrame(mean_field_proportion_vectors, columns=[f'Clase {i}' for i in range(num_states)])
    df_mean_field.index.rename("Número de iteración", inplace=True)
    st.dataframe(df_mean_field)  

    st.markdown("""
        ---
        **Explicación:**  
                
        En esta gráfica se pueden apreciar los valores de las configuraciones en la evolución del sistema original, comparado con las configuraciones
        obtenidas a través del modelo de campo medio. La evolución en campo medio permanece bastante estable, pero similar a la configuración del modelo original.
    """)


#endregion