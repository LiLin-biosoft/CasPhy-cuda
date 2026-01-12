#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <set>
#include <cctype>
#include <iomanip>

// Encoding function
uint32_t encode_char(char c) {
    switch (c) {
    case 'A': case 'a': return 0x01;
    case 'T': case 't': return 0x02;
    case 'U': case 'u': return 0x02;
    case 'C': case 'c': return 0x04;
    case 'G': case 'g': return 0x08;
    case 'W': case 'w': return 0x03; //AT
    case 'M': case 'm': return 0x05; //AC
    case 'R': case 'r': return 0x09; //AG
    case 'Y': case 'y': return 0x06; //TC
    case 'K': case 'k': return 0x0A; //TG
    case 'S': case 's': return 0x0C; //CG
    case 'H': case 'h': return 0x07; //ATC
    case 'D': case 'd': return 0x0B; //ATG
    case 'V': case 'v': return 0x0D; //ACG
    case 'B': case 'b': return 0x0E; //TCG
    case 'N': case 'n': return 0x0F; //ATCG
    default: return 0x10;
    }
}

// Function to count valid bases (non '-', non '.', non 'N')
int count_valid_bases(const std::string& sequence) {
    int count = 0;
    for (char c : sequence) {
        char upper_c = std::toupper(c);
        if (upper_c != '-' && upper_c != '.' && upper_c != 'N') {
            count++;
        }
    }
    return count;
}

// Function to check if a position contains only one type of base across all sequences
std::vector<bool> find_variable_positions(const std::vector<std::string>& sequences) {
    if (sequences.empty()) return std::vector<bool>();
    
    size_t seq_length = sequences[0].length();
    std::vector<bool> is_variable(seq_length, false);
    
    // Check each position across all sequences
    for (size_t pos = 0; pos < seq_length; ++pos) {
        unsigned int cnt[5] = {0};
        unsigned int nChecked=0;
        for (const auto& seq : sequences) {
            if (pos < seq.length()) {
                char base = std::toupper(seq[pos]);
			    switch (base) {
			    case 'A': case 'a': cnt[0]++;break;
			    case 'T': case 't': cnt[1]++;break;
			    case 'U': case 'u': cnt[1]++;break;
			    case 'C': case 'c': cnt[2]++;break;
			    case 'G': case 'g': cnt[3]++;break;
			    case 'W': case 'w': cnt[0]++;cnt[1]++;break; //AT
			    case 'M': case 'm': cnt[0]++;cnt[2]++;break; //AC
			    case 'R': case 'r': cnt[0]++;cnt[3]++;break; //AG
			    case 'Y': case 'y': cnt[1]++;cnt[2]++;break; //TC
			    case 'K': case 'k': cnt[1]++;cnt[3]++;break; //TG
			    case 'S': case 's': cnt[2]++;cnt[3]++;break; //CG
			    case 'H': case 'h': cnt[0]++;cnt[1]++;cnt[2]++;break; //ATC
			    case 'D': case 'd': cnt[0]++;cnt[1]++;cnt[3]++;break; //ATG
			    case 'V': case 'v': cnt[0]++;cnt[2]++;cnt[3]++;break; //ACG
			    case 'B': case 'b': cnt[1]++;cnt[2]++;cnt[3]++;break; //TCG
			    case 'N': case 'n': cnt[0]++;cnt[1]++;cnt[2]++;cnt[3]++;break; //ATCG
			    default: cnt[4]++;break;
			    }
			    nChecked++;
			    for(unsigned int cntIdx=0;cntIdx<4;++cntIdx){
			    	if(cnt[cntIdx]>0&&cnt[cntIdx]<nChecked){
			    		is_variable[pos] = true;
			    		break;
					}
				}
                if (is_variable[pos]) {
                    break;
                }
            }
        }
    }
    return is_variable;
}

// Remove non-variable positions from sequences
std::vector<std::string> filter_variable_positions(const std::vector<std::string>& sequences, 
                                                  const std::vector<bool>& variable_positions,
                                                  int& removed_count) {
    std::vector<std::string> filtered_sequences;
    removed_count = 0;
    
    for (size_t pos = 0; pos < variable_positions.size(); ++pos) {
        if (!variable_positions[pos]) {
            removed_count++;
        }
    }
    
    for (const auto& seq : sequences) {
        std::string filtered_seq;
        for (size_t pos = 0; pos < seq.length(); ++pos) {
            if (pos < variable_positions.size() && variable_positions[pos]) {
                filtered_seq += seq[pos];
            }
        }
        filtered_sequences.push_back(filtered_seq);
    }
    
    return filtered_sequences;
}

// Read FASTA file and extract sequence names and sequences
void read_fasta(const std::string& filename, 
                std::vector<std::string>& sequence_names, 
                std::vector<std::string>& sequences,
                std::vector<int>& valid_base_counts) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string current_name;
    std::string current_sequence;
    std::string line;
    
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        
        // Check if it's a header line
        if (line[0] == '>') {
            // Save previous sequence if exists
            if (!current_sequence.empty()) {
                sequences.push_back(current_sequence);
                sequence_names.push_back(current_name);
                valid_base_counts.push_back(count_valid_bases(current_sequence));
                current_sequence.clear();
            }
            
            // Extract sequence name (remove '>' and trim)
            current_name = line.substr(1);
            size_t first_non_space = current_name.find_first_not_of(" \t");
            size_t first_space = current_name.find_first_of(" \t");
            
            if (first_non_space != std::string::npos) {
                if (first_space != std::string::npos) {
                    current_name = current_name.substr(first_non_space, first_space - first_non_space);
                } else {
                    current_name = current_name.substr(first_non_space);
                }
            }
        } else {
            // Remove whitespace characters from sequence line
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            current_sequence += line;
        }
    }
    
    // Save the last sequence
    if (!current_sequence.empty()) {
        sequences.push_back(current_sequence);
        sequence_names.push_back(current_name);
        valid_base_counts.push_back(count_valid_bases(current_sequence));
    }
    
    if (sequences.empty()) {
        throw std::runtime_error("No valid sequences found in file");
    }
    
    // Check if all sequences have the same length
    size_t first_length = sequences[0].length();
    for (size_t i = 1; i < sequences.size(); ++i) {
        if (sequences[i].length() != first_length) {
            throw std::runtime_error("All sequences must have the same length for variant filtering");
        }
    }
}

// Encode entire sequence - 8 bases per 32-bit integer
std::vector<uint32_t> encode_sequence(const std::string& sequence) {
    size_t n = sequence.size();
    if (n == 0) return std::vector<uint32_t>();
    
    // Calculate output length: ceil(n / 4)
    size_t out_len = (n + 3) / 4;
    std::vector<uint32_t> encoded(out_len);
    
    for (size_t i = 0; i < out_len; ++i) {
        uint32_t v = 0;
        for (int j = 0; j < 4; ++j) {
            size_t pos = i * 4 + j;
            if (pos < n) {
                uint32_t code = encode_char(sequence[pos]);
                v |= (code << (j * 8));
            }
        }
        encoded[i] = v;
    }
    
    return encoded;
}

// Save to binary file
void save_binary(const std::vector<uint32_t>& encoded_data, 
                 const std::string& output_filename) {
    std::ofstream outfile(output_filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create output file: " + output_filename);
    }
    
    // Write encoded data
    outfile.write(reinterpret_cast<const char*>(encoded_data.data()), 
                 encoded_data.size() * sizeof(uint32_t));
}

// Save sequence names and valid base counts to text file
void save_sequence_names(const std::vector<std::string>& sequence_names, 
                        const std::vector<int>& valid_base_counts,
                        const std::string& output_filename) {
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create output file: " + output_filename);
    }
    
    // Write header
    outfile << "SequenceName,ValidBaseCount\n";
    
    for (size_t i = 0; i < sequence_names.size(); ++i) {
        outfile << sequence_names[i] << "," << valid_base_counts[i] << '\n';
    }
}

// Save variable positions information
void save_variable_positions_info(const std::vector<bool>& variable_positions,
                                 const std::string& output_filename) {
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create output file: " + output_filename);
    }
    
    outfile << "Position,IsVariable\n";
    for (size_t i = 0; i < variable_positions.size(); ++i) {
        outfile << i << "," << (variable_positions[i] ? "Yes" : "No") << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <input FASTA file> <output binary file> <output names file> <output positions info file>" << std::endl;
        std::cout << "Example: " << argv[0] << " input.fasta sequences.bin names.txt positions.csv" << std::endl;
        std::cout << "Note: All sequences must have the same length for variant filtering" << std::endl;
        return 1;
    }
    
    std::string input_filename = argv[1];
    std::string output_binary_filename = argv[2];
    std::string output_names_filename = argv[3];
    std::string output_positions_filename = argv[4];
    
    try {
        // Read FASTA file
        std::cout << "Reading FASTA file: " << input_filename << std::endl;
        std::vector<std::string> sequence_names;
        std::vector<std::string> sequences;
        std::vector<int> valid_base_counts;
        read_fasta(input_filename, sequence_names, sequences, valid_base_counts);
        
        std::cout << "Found " << sequences.size() << " sequences" << std::endl;
        std::cout << "Original sequence length: " << sequences[0].length() << " bases" << std::endl;
        
        // Display valid base counts for each sequence
        std::cout << "\nValid base counts for each sequence:" << std::endl;
        for (size_t i = 0; i < sequence_names.size(); ++i) {
            std::cout << "  " << sequence_names[i] << ": " << valid_base_counts[i] 
                      << " valid bases (" << std::fixed << std::setprecision(1) 
                      << (valid_base_counts[i] * 100.0 / sequences[i].length()) << "%)" << std::endl;
        }
        
        // Find variable positions
        std::cout << "\nAnalyzing variable positions..." << std::endl;
        std::vector<bool> variable_positions = find_variable_positions(sequences);
        
        // Filter out non-variable positions
        int removed_count;
        std::vector<std::string> filtered_sequences = filter_variable_positions(sequences, variable_positions, removed_count);
        
        std::cout << "Removed " << removed_count << " non-variable positions" << std::endl;
        std::cout << "Filtered sequence length: " << filtered_sequences[0].length() << " bases" << std::endl;
        std::cout << "Retained " << (sequences[0].length() - removed_count) << " variable positions" << std::endl;
        
        // Save variable positions information
        std::cout << "Saving positions information to: " << output_positions_filename << std::endl;
        save_variable_positions_info(variable_positions, output_positions_filename);
        
        // Process all filtered sequences
        std::vector<std::vector<uint32_t>> all_encoded_data;
        
        for (size_t i = 0; i < filtered_sequences.size(); ++i) {
            std::cout << "Processing sequence " << (i + 1) << ": " << sequence_names[i] << std::endl;
            std::cout << "  Filtered length: " << filtered_sequences[i].length() << " bases" << std::endl;
            
            std::vector<uint32_t> encoded_data = encode_sequence(filtered_sequences[i]);
            
            std::cout << "  Encoded data size: " << encoded_data.size() << " 32-bit integers" << std::endl;
            std::cout << "  Compression ratio: " << filtered_sequences[i].length() << " bases -> " 
                      << encoded_data.size() << " integers (8:1)" << std::endl;
            
            all_encoded_data.push_back(encoded_data);
        }
        
        // Save sequence names and valid base counts to text file
        std::cout << "Saving sequence names and valid base counts to: " << output_names_filename << std::endl;
        save_sequence_names(sequence_names, valid_base_counts, output_names_filename);
        
        // Save encoded data to binary file
        std::cout << "Saving encoded data to: " << output_binary_filename << std::endl;
        
        std::ofstream outfile(output_binary_filename, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Cannot create output file: " + output_binary_filename);
        }
        
        // Write metadata header
        uint32_t num_sequences = filtered_sequences.size();
        uint32_t encoded_length = all_encoded_data[0].size();
        
        outfile.write(reinterpret_cast<const char*>(&num_sequences), sizeof(num_sequences));
        outfile.write(reinterpret_cast<const char*>(&encoded_length), sizeof(encoded_length));
        
        // Write each sequence's encoded data
        for (size_t i = 0; i < filtered_sequences.size(); ++i) {
            outfile.write(reinterpret_cast<const char*>(all_encoded_data[i].data()), 
                         all_encoded_data[i].size() * sizeof(uint32_t));
        }
        
        std::cout << "\nConversion completed!" << std::endl;
        std::cout << "Sequence names and valid base counts saved to: " << output_names_filename << std::endl;
        std::cout << "Encoded data saved to: " << output_binary_filename << std::endl;
        std::cout << "Position information saved to: " << output_positions_filename << std::endl;
        std::cout << "Binary file structure:" << std::endl;
        std::cout << "  - Header: " << sizeof(uint32_t) * 2 << " bytes (num_seqs, enc_len)" << std::endl;
        std::cout << "  - Data: " << (all_encoded_data.size() * all_encoded_data[0].size() * sizeof(uint32_t)) 
                  << " bytes (" << all_encoded_data.size() << " sequences * " << all_encoded_data[0].size() << " integers)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
