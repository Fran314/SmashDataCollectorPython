#include <string>
#include <list>
#include <opencv2/opencv.hpp>

using namespace std;

class match
{
private:
    string date;
    string first_scrn_file, second_scrn_file;
public:
    match(string file_name, string first_scrn_file, string second_scrn_file);
    ~match();

    void load_images();
    void unload_images();

    void init();

    cv::Mat first_scrn, second_scrn;
    int n_players;

    //--- FIRST SCREENSHOT DATA ---//
    int *types;
    int *characters;
    int *positions;
    int *times;

    //--- SECOND SCREENSHOT DATA ---//
    list<int> *falls;
    int *given_dmg;
    int *taken_dmg;
};

match::match(string file_name, string _first_scrn_file, string _second_scrn_file)
{
    first_scrn_file = _first_scrn_file;
    second_scrn_file = _second_scrn_file;

    date = file_name.substr(6, 2) + '/' + file_name.substr(4, 2) + '/' + file_name.substr(0, 4);
}

void match::init()
{
    types = new int[n_players];
    characters = new int[n_players];
    positions = new int[n_players];
    times = new int[n_players];

    falls = new list<int>[n_players];
    given_dmg = new int[n_players];
    taken_dmg = new int[n_players];
}

void match::load_images()
{
    first_scrn = cv::imread(first_scrn_file);
    second_scrn = cv::imread(second_scrn_file);
}

void match::unload_images()
{
    first_scrn.release();
    second_scrn.release();
}

match::~match()
{
    unload_images();
}